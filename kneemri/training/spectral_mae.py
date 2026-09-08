"""Masked image modelling with a spectral reconstruction term, for adapting the encoder to knee MRI.

THE GATE THIS STANDS ON IS ALREADY MEASURED. `kneemri.evaluation.encoder_drift` reported that in every
DINOv2 member we have shipped, blocks 0-5 drift EXACTLY 0.0000 — `--unfreeze-depth 6` freezes them — and
the most-adapted block of our best member moved 2.1% of its own norm. Supervision here re-weights a
natural-image representation rather than rebuilding one, and it is arithmetically incapable of touching
half the encoder. On-domain pretraining writes into precisely those blocks, so the usual objection to
domain-adaptive pretraining (the fine-tune undoes it) does not apply to the frozen half at all.

WHY RECONSTRUCTION RATHER THAN SELF-DISTILLATION, which is what knee-c5n originally proposed. On 3D
brain MRI, masked autoencoding beats DINO-style and JEPA-style objectives, and the margin is widest in
the FROZEN-encoder regime that half our encoder is permanently in. Our own ledger says the same thing
from the other side: DINOv3 — strictly better natural-image self-distillation — scored ~0.04 WORSE than
DINOv2 here, with the deficit in focal findings. More of the objective that already disappoints is not
the change worth making.

THE SPECTRAL TERM IS THE POINT, not decoration. A plain pixel MSE is dominated by low frequencies —
bulk intensity and large smooth structure — which is the half of the image our labels are already good
at (effusion, synovitis, Baker's, the large diffuse findings). The half we measurably fail on is focal
and small: menisci, cruciates, collaterals, patellofemoral cartilage. So the reconstruction is scored a
second time in the frequency domain with the low frequencies REMOVED, which prices exactly the detail a
pixel loss is free to blur away.

The masking hook is the backbone's own: `Dinov2Model.forward(pixel_values, bool_masked_pos)` substitutes
a learned mask token at the embedding layer. Reaching through `backbone.embeddings` by hand would have
worked for one architecture and silently mis-indexed the register tokens of the next.
"""

from dataclasses import dataclass
from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from kneemri.models.backbone import IMAGENET, Normalization

BYTE_RANGE = 255.0
CHANNELS = 3
"""What the encoder expects. The cache is single-channel, so one slice is replicated across all three —
which is also why the decoder predicts ONE channel: predicting three identical ones would triple the
output width to score the same pixel three times."""

_VARIANCE_FLOOR = 1.0e-6
"""Added under the square root of a patch's own standard deviation before dividing by it.

Per-patch target normalisation is MAE's, and it exists so that a flat background patch and a textured
joint patch contribute comparably instead of the loss being an intensity histogram. A patch of pure
background has variance zero, and without a floor its normalised target is a division by zero — which
arrives as NaN in the loss several hundred steps later, with nothing left pointing at the cause."""


@dataclass(frozen=True)
class SpectralMaskConfig:
    """What the objective does to an image, and how the two loss terms are weighed against each other."""

    mask_ratio: float = 0.6
    """The share of patches replaced by the mask token.

    0.6 rather than MAE's 0.75 because this encoder sees the FULL token sequence: MAE can afford to
    discard 75% because its encoder never reads the masked positions at all, which both cheapens the
    forward pass and removes the easy solution of copying a neighbour. Running the full sequence — the
    only shape a stock `Dinov2Model` offers — makes a very high ratio pay full compute for a harder task
    than the decoder can use. 0.6 is the ratio the full-sequence line (SimMIM) settled on."""

    spectral_weight: float = 1.0
    """How much the high-frequency term counts against the pixel term.

    THE PAPER DOES NOT STATE ITS WEIGHT, so this is the arm's free variable and not a replicated
    constant. 1.0 is the neutral starting point rather than a tuned one: both terms are mean squared
    errors over quantities of order one (targets are per-patch standardised, spectra are `log1p` of a
    magnitude), so equal weight is the assumption that neither term is a priori more important. Say so
    in the write-up rather than reporting it as the paper's setting."""

    highpass: float = 0.125
    """The radius, as a fraction of the spectrum's half-diagonal, inside which frequencies are discarded.

    Everything within this radius of DC is zeroed after `fftshift`, so the term scores only the outer
    ~98% of the frequency plane by area. 0.125 puts the cutoff at a wavelength of about 8 patches — at
    336 px and patch 14 that is roughly 112 px, comfortably larger than any structure we are failing on
    (a meniscal horn is a few millimetres) and comfortably smaller than the bulk-intensity variation the
    pixel term already handles."""

    patch: int = 14
    """The encoder's patch side. DINOv2 is 14 and DINOv3 is 16; a mismatch would tile the image on a grid
    the position embeddings were never trained for, so it comes from the `Encoder` record, not a default."""


class SpectralMae(nn.Module):
    """A backbone, a linear pixel decoder, and the two-term reconstruction loss that trains them.

    The decoder is one `Linear` on purpose. MAE's eight-block transformer decoder exists to make a
    75%-masked, encoder-drops-masked-tokens setup solvable; on a full-sequence encoder the reconstruction
    head can be linear without loss, and every parameter spent there is a parameter not spent on the
    thing we intend to keep. Only the encoder is saved at the end — the decoder is scaffolding.
    """

    mean: Tensor
    std: Tensor

    def __init__(
        self,
        backbone: nn.Module,
        dim: int,
        config: SpectralMaskConfig,
        normalization: Normalization = IMAGENET,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.decoder = nn.Linear(dim, config.patch * config.patch)
        # The trunk's OWN pretraining constants, for the reason `StudyModel` states: pretraining a
        # checkpoint through somebody else's normalisation is a silent handicap, and this path is the
        # one where the trunk is the entire subject of the run.
        self.register_buffer("mean", torch.tensor(normalization.mean).view(1, CHANNELS, 1, 1))
        self.register_buffer("std", torch.tensor(normalization.std).view(1, CHANNELS, 1, 1))

    def patchify(self, images: Float[Tensor, "b 1 h w"]) -> Float[Tensor, "b n q"]:
        """One row per patch, in the raster order the backbone emits its patch tokens in.

        Raster order is asserted by construction rather than assumed: the reshape below walks rows of
        patches before columns, which is the order a vision transformer flattens its grid in, and a
        transposed target would train the decoder to predict a different patch from the one it was
        handed with no error anywhere to notice it.
        """
        patch = self.config.patch
        batch, _, height, width = images.shape
        grid_h, grid_w = height // patch, width // patch
        tiled = images.reshape(batch, 1, grid_h, patch, grid_w, patch)
        return tiled.permute(0, 2, 4, 3, 5, 1).reshape(batch, grid_h * grid_w, patch * patch)

    def unpatchify(self, patches: Float[Tensor, "b n q"], grid: int) -> Float[Tensor, "b 1 h w"]:
        """The inverse of `patchify`, so a reconstruction can be scored as an image rather than as rows.

        The spectral term needs this: a Fourier transform of a stack of independent patches measures the
        frequency content of a patch, not of the anatomy, and the structures at issue span several
        patches.
        """
        patch = self.config.patch
        batch = patches.shape[0]
        tiled = patches.reshape(batch, grid, grid, patch, patch, 1)
        return tiled.permute(0, 5, 1, 3, 2, 4).reshape(batch, 1, grid * patch, grid * patch)

    def sample_mask(
        self, batch: int, tokens: int, device: torch.device, generator: torch.Generator
    ) -> Bool[Tensor, "b n"]:
        """Which patches are hidden, with the SAME count hidden in every sample of the batch.

        A per-sample Bernoulli draw would give each sample a different number of masked patches and so a
        different loss denominator, which turns the batch mean into a weighted average whose weights are
        noise. Ranking a uniform draw and taking a fixed-size prefix keeps the positions random and the
        count exact.

        DRAWN ON THE CPU AND MOVED, whatever the model's device is. `torch.rand` refuses a CPU generator
        for a CUDA tensor outright, and the alternative — a second, CUDA-side generator — would mean the
        run's seed no longer determines the masking, which is the reproducibility this arm's A/B rests
        on. The mask is a few hundred booleans per sample; the transfer is not the cost here.
        """
        hidden = max(1, round(tokens * self.config.mask_ratio))
        order = torch.rand(batch, tokens, generator=generator).argsort(dim=1).to(device)
        mask = torch.zeros(batch, tokens, dtype=torch.bool, device=device)
        return mask.scatter(1, order[:, :hidden], value=True)

    def encode(self, images: Float[Tensor, "b 1 h w"], mask: Bool[Tensor, "b n"]) -> Float[Tensor, "b n d"]:
        """Patch tokens for a masked image, with the prefix tokens dropped.

        The prefix — one class token for DINOv2, plus four registers for DINOv3 — is dropped by counting
        BACK from the end rather than forward from the start, because only the patch count is known here
        and the prefix width is the backbone's business.
        """
        expanded = images.expand(-1, CHANNELS, -1, -1)
        normalised = (expanded - self.mean) / self.std
        hidden = self.backbone(pixel_values=normalised, bool_masked_pos=mask).last_hidden_state
        return hidden[:, hidden.shape[1] - mask.shape[1] :]

    def pixel_loss(
        self, predicted: Float[Tensor, "b n q"], target: Float[Tensor, "b n q"], mask: Bool[Tensor, "b n"]
    ) -> Float[Tensor, ""]:
        """Mean squared error over the HIDDEN patches only, against per-patch standardised targets.

        Scoring the visible patches too would let the model spend capacity on copying its own input,
        which is the one part of the task that carries no information about the anatomy.
        """
        centred = target - target.mean(dim=-1, keepdim=True)
        standardised = centred / (target.var(dim=-1, keepdim=True) + _VARIANCE_FLOOR).sqrt()
        error = (predicted - standardised).pow(2).mean(dim=-1)
        return (error * mask).sum() / mask.sum()

    def restore(self, predicted: Float[Tensor, "b n q"], target: Float[Tensor, "b n q"]) -> Float[Tensor, "b n q"]:
        """Put a standardised prediction back into the target's own pixel units.

        `pixel_loss` scores the decoder against per-patch standardised targets, so the decoder's output
        lives at mean 0 and unit variance while a patch of the image lives in [0, 1] at a standard
        deviation of order 0.15. Merging the two without this step builds a mosaic in two different
        units, and the spectral term then measures the block discontinuity between them rather than the
        detail the model invented. That is not a scaling nicety: the two terms want opposite outputs
        from the same decoder, the spectral term carries the larger gradient, and the pixel term parks
        at the 1.0 that predicting the patch mean scores. Both eight-epoch runs did exactly that.
        """
        deviation = (target.var(dim=-1, keepdim=True) + _VARIANCE_FLOOR).sqrt()
        return predicted * deviation + target.mean(dim=-1, keepdim=True)

    def highpass(self, images: Float[Tensor, "b 1 h w"]) -> Float[Tensor, "b 1 h w"]:
        """The log-magnitude spectrum with the low frequencies removed.

        `log1p` rather than a raw magnitude because a natural image's spectrum spans several orders of
        magnitude and an unlogged MSE would be a measurement of the DC term and its immediate
        neighbours — the exact frequencies this then deletes.
        """
        spectrum = torch.fft.fftshift(torch.fft.fft2(images.float()), dim=(-2, -1))
        magnitude = torch.log1p(spectrum.abs())
        height, width = images.shape[-2:]
        rows = torch.arange(height, device=images.device) - height / 2
        columns = torch.arange(width, device=images.device) - width / 2
        radius = (rows.view(-1, 1) ** 2 + columns.view(1, -1) ** 2).sqrt()
        cutoff = self.config.highpass * radius.max()
        return magnitude * (radius > cutoff)

    def spectral_loss(
        self, reconstruction: Float[Tensor, "b 1 h w"], original: Float[Tensor, "b 1 h w"]
    ) -> Float[Tensor, ""]:
        """How far the reconstruction's fine detail is from the original's, measured in frequency.

        Scored over the WHOLE image rather than the hidden patches, because a frequency is not local:
        there is no sub-spectrum belonging to one patch, and a reconstruction that is sharp inside the
        masked region and discontinuous at its border is not the sharp reconstruction we asked for.
        """
        return (self.highpass(reconstruction) - self.highpass(original)).pow(2).mean()

    @override
    def forward(
        self, images: Float[Tensor, "b 1 h w"], mask: Bool[Tensor, "b n"]
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
        """The total loss and both of its terms, so a run can log which one is actually moving.

        Both terms are returned rather than only their sum for the reason the ledger keeps re-learning:
        a combined number that improves says nothing about whether the mechanism being claimed is the
        one doing the work, and the spectral term IS the claim here.
        """
        target = self.patchify(images)
        predicted = self.decoder(self.encode(images, mask))
        pixels = self.pixel_loss(predicted, target, mask)
        grid = images.shape[-1] // self.config.patch
        # The visible patches are restored from the ORIGINAL rather than predicted, so the spectral term
        # scores the model on the detail it had to invent and not on how faithfully it copied what it was
        # shown. Detaching them additionally keeps the gradient off the copy. The predicted patches pass
        # through `restore` first: they arrive standardised, the visible ones do not, and merging the two
        # unconverted is what made both eight-epoch runs optimise the spectrum and abandon the pixels.
        merged = torch.where(mask.unsqueeze(-1), self.restore(predicted, target.detach()), target.detach())
        spectral = self.spectral_loss(self.unpatchify(merged, grid), images)
        return pixels + self.config.spectral_weight * spectral, pixels, spectral

    def encoder_state(self) -> dict[str, Tensor]:
        """The adapted backbone alone, keyed exactly as `AutoModel.from_pretrained` keys it.

        Exactly, because the whole point is that the supervised trainer can load this in place of the
        stock weights without touching any other part of its setup — and because `encoder_drift` then
        measures this run against the same stock reference on the same key names. The decoder and the
        normalisation buffers are dropped: they are scaffolding, and a state dict carrying extra keys is
        a state dict `load_state_dict` refuses.
        """
        return {name: value.detach().cpu().clone() for name, value in self.backbone.state_dict().items()}
