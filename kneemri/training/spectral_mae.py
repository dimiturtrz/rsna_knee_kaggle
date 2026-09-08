"""Masked image modelling with a spectral reconstruction term, for adapting the encoder to knee MRI.

THE GATE THIS STANDS ON IS ALREADY MEASURED. `kneemri.evaluation.encoder_drift` reported that in every
DINOv2 member we have shipped, blocks 0-5 drift EXACTLY 0.0000 — `unfreeze_last: 6` freezes them — and
the most-adapted block of our best member moved 2.1% of its own norm. Supervision here re-weights a
natural-image representation rather than rebuilding one, and it is arithmetically incapable of touching
half the encoder. On-domain pretraining writes into precisely those blocks, so the usual objection to
domain-adaptive pretraining (the fine-tune undoes it) does not apply to the frozen half at all.

WHY RECONSTRUCTION RATHER THAN SELF-DISTILLATION — AND BOTH REASONS TURNED OUT TO BE DEFECTS. This
docstring argued for pixels on two claims, and researcher D249 refuted both. The first, that "on 3D brain
MRI masked autoencoding beats DINO-style and JEPA-style objectives", carried NO citation and none was
found. The second compared two released CHECKPOINTS — DINOv3 scoring ~0.04 below DINOv2 here — to license
a claim about self-distillation as an OBJECTIVE; those checkpoints differ in data, scale, recipe and
registers, so the inference does not run.

The measured direction is the other one. CAPI (Darcet et al., arXiv 2502.08769) on ViT-L/IN1k frozen
probes: MAE 79.4, iBOT 80.9, CAPI 82.9, with pixel targets called out as "too low-level". And our own
`ortholarge` pair — a LATENT continuation on 1.25M knee images — is +0.0194 / +0.0140 at matched recipe
where this objective's arm ran -0.0885. `kneemri.training.latent_mim` is the sibling that changes only
the target; this module is kept as its control, not as the recommended path.

WHAT ACTUALLY HAPPENED TO THE ARM, which is a third story neither the pixel nor the latent argument
predicts: researcher D251 measured `adapt_small_e8_unitfix` and found blocks 8-11 within 0.11 nats of
`ln(256)` — uniform attention, no routing at all — where stock sits at 4.69. `unfreeze_last: 6` opens
exactly the collapsed blocks, so the fine-tune spent twelve epochs rebuilding attention. The -0.0885 is a
fact about a damaged initialisation and NOT a data point on pixel-vs-latent. `kneemri.training.
attention_health` is the gate that now ends such a run at epoch one.

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
from kneemri.training.masked_objective import MaskConfig, MaskedObjective

_VARIANCE_FLOOR = 1.0e-6
"""Added under the square root of a patch's own standard deviation before dividing by it.

Per-patch target normalisation is MAE's, and it exists so that a flat background patch and a textured
joint patch contribute comparably instead of the loss being an intensity histogram. A patch of pure
background has variance zero, and without a floor its normalised target is a division by zero — which
arrives as NaN in the loss several hundred steps later, with nothing left pointing at the cause."""


@dataclass(frozen=True)
class SpectralMaskConfig(MaskConfig):
    """What this objective adds to the shared masking: how the two loss terms are weighed, and where the
    frequency cutoff sits."""

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


class SpectralMae(MaskedObjective[SpectralMaskConfig]):
    """A backbone, a linear pixel decoder, and the two-term reconstruction loss that trains them.

    The decoder is one `Linear` on purpose. MAE's eight-block transformer decoder exists to make a
    75%-masked, encoder-drops-masked-tokens setup solvable; on a full-sequence encoder the reconstruction
    head can be linear without loss, and every parameter spent there is a parameter not spent on the
    thing we intend to keep. Only the encoder is saved at the end — the decoder is scaffolding.
    """

    def __init__(
        self,
        backbone: nn.Module,
        dim: int,
        config: SpectralMaskConfig,
        normalization: Normalization = IMAGENET,
    ) -> None:
        super().__init__(backbone, config, normalization)
        self.decoder = nn.Linear(dim, config.patch * config.patch)

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
    def term_names(self) -> tuple[str, str]:
        return ("pixel", "spectral")

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
