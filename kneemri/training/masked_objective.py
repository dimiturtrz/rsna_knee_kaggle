"""What every masked-image objective on this encoder shares: the mask, the forward, and what is saved.

WHY THIS EXISTS AS A BASE RATHER THAN A SECOND COPY. `SpectralMae` was the only masked objective here,
so its masking, its normalisation and its "save the backbone alone" rule lived inside it. The axis that
turned out to matter is not masking at all — DINOv2's own pretraining is DINO + iBOT, so the encoder is
ALREADY a masked-image model — it is what the masked positions are regressed AGAINST: pixels, or the
features of a teacher. Everything except that target is common, and a second objective that copied it
would be a second place for the prefix-trimming and the per-patch masking to drift.

THE PREFIX IS TRIMMED BY COUNTING BACK FROM THE END, here and nowhere else. DINOv2 prepends one class
token and DINOv3 four registers, and only the patch count is known at this layer.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from kneemri.models.backbone import IMAGENET, Normalization

BYTE_RANGE = 255.0
CHANNELS = 3
"""What the encoder expects. The cache is single-channel, so one slice is replicated across all three."""


@dataclass(frozen=True)
class MaskConfig:
    """The two knobs that decide what an objective is shown, independent of what it is scored on."""

    mask_ratio: float = 0.6
    """The share of patches replaced by the mask token.

    0.6 rather than MAE's 0.75 because this encoder sees the FULL token sequence: MAE can afford to
    discard 75% because its encoder never reads the masked positions at all. 0.6 is the ratio the
    full-sequence line (SimMIM) settled on."""

    patch: int = 14
    """The encoder's patch side. DINOv2 is 14 and DINOv3 is 16; it comes from the `Encoder` record."""


class MaskedObjective[C: MaskConfig](nn.Module):
    """A backbone, a masking rule, and whatever loss a subclass scores the masked positions with.

    Generic in its config because a subclass reads knobs the base has never heard of, and narrowing an
    inherited attribute by re-annotation is the one thing a type checker will not allow. The parameter
    says the narrowing is the DESIGN — one objective, one config — and `config` is exposed READ-ONLY so
    the parameter is covariant: a `MaskedObjective[LatentMimConfig]` is then usable wherever the
    catalogue promises a `MaskedObjective[MaskConfig]`, which a mutable attribute would forbid. That is the
    spelling every call site uses when it does not care which objective it holds (a PEP 696 default would
    say it in one word, and needs Python 3.13).
    """

    mean: Tensor
    std: Tensor

    def __init__(
        self,
        backbone: nn.Module,
        config: C,
        normalization: Normalization = IMAGENET,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self._config = config
        # The trunk's OWN pretraining constants: pretraining a checkpoint through somebody else's
        # normalisation is a silent handicap, and here the trunk is the entire subject of the run.
        self.register_buffer("mean", torch.tensor(normalization.mean).view(1, CHANNELS, 1, 1))
        self.register_buffer("std", torch.tensor(normalization.std).view(1, CHANNELS, 1, 1))

    @property
    def config(self) -> C:
        """The knobs this objective was built with. Read-only: see the covariance note on the class."""
        return self._config

    def sample_mask(
        self, batch: int, tokens: int, device: torch.device, generator: torch.Generator
    ) -> Bool[Tensor, "b n"]:
        """Which patches are hidden, with the SAME count hidden in every sample of the batch.

        A per-sample Bernoulli draw would give each sample a different loss denominator, which turns the
        batch mean into a weighted average whose weights are noise. Ranking a uniform draw and taking a
        fixed-size prefix keeps the positions random and the count exact.

        DRAWN ON THE CPU AND MOVED, whatever the model's device is: `torch.rand` refuses a CPU generator
        for a CUDA tensor, and a second CUDA-side generator would mean the run's seed no longer
        determines the masking.
        """
        hidden = max(1, round(tokens * self.config.mask_ratio))
        order = torch.rand(batch, tokens, generator=generator).argsort(dim=1).to(device)
        mask = torch.zeros(batch, tokens, dtype=torch.bool, device=device)
        return mask.scatter(1, order[:, :hidden], value=True)

    def normalise(self, images: Float[Tensor, "b 1 h w"]) -> Float[Tensor, "b c h w"]:
        """One cached slice, replicated to the channels the trunk expects and put in its own units."""
        return (images.expand(-1, CHANNELS, -1, -1) - self.mean) / self.std

    def patches(self, hidden: Float[Tensor, "b t d"], tokens: int) -> Float[Tensor, "b n d"]:
        """The patch tokens alone, counted BACK from the end.

        Back from the end because only the patch count is known at this layer: DINOv2 prepends one class
        token and DINOv3 four registers, and a forward index would work for one architecture and
        silently mis-align the other by four positions with no error anywhere to notice it.
        """
        return hidden[:, hidden.shape[1] - tokens :]

    def encode(self, images: Float[Tensor, "b 1 h w"], mask: Bool[Tensor, "b n"]) -> Float[Tensor, "b n d"]:
        """Patch tokens for a masked image, with the prefix tokens dropped.

        The masking hook is the backbone's own: `Dinov2Model.forward(pixel_values, bool_masked_pos)`
        substitutes a learned mask token at the embedding layer. Reaching through `backbone.embeddings`
        by hand would work for one architecture and silently mis-index the registers of the next.
        """
        hidden = self.backbone(pixel_values=self.normalise(images), bool_masked_pos=mask).last_hidden_state
        return self.patches(hidden, mask.shape[1])

    def plan(self, updates: int) -> None:
        """Whatever the objective needs to know about the run's LENGTH before it starts. Nothing, here.

        It exists because a teacher-based objective's momentum is only meaningful against the number of
        updates it will see, and that number is known after the trainer is built — so it cannot be a
        constructor argument without making the objective depend on the trainer that consumes it.
        """
        return

    def after_step(self) -> None:
        """Whatever the objective owes the optimiser step it did not take part in. Nothing, by default.

        A teacher-based objective updates its teacher here rather than inside `forward`, because an EMA
        that moved during the forward pass would make the target a function of the batch it is scoring.
        """
        return

    @abstractmethod
    def term_names(self) -> tuple[str, str]:
        """What the two reported terms ARE, so a log line names its own numbers.

        Both terms are reported rather than only their sum for the reason the ledger keeps re-learning:
        a combined number that improves says nothing about whether the claimed mechanism did the work.
        """

    @abstractmethod
    @override
    def forward(
        self, images: Float[Tensor, "b 1 h w"], mask: Bool[Tensor, "b n"]
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
        """The total loss and both of its reported terms, in the order `term_names` names them."""

    def encoder_state(self) -> dict[str, Tensor]:
        """The adapted backbone alone, keyed exactly as `AutoModel.from_pretrained` keys it.

        Exactly, because the supervised trainer must load this in place of the stock weights without
        touching any other part of its setup — and because `encoder_drift` then measures this run
        against the same stock reference on the same key names. Decoders, projections, teachers and the
        normalisation buffers are scaffolding and are dropped.
        """
        return {name: value.detach().cpu().clone() for name, value in self.backbone.state_dict().items()}
