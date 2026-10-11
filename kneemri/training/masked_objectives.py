"""What an unlabelled run can be scored against, and the dispatch `--objective` resolves through.

Plural, and beside `masked_objective.py`: this is the CATALOGUE of `MaskedObjective` builders, not to be
read as a sibling of `objective.py`, which is the SUPERVISED loss and a different subject entirely.

Its own module rather than a second class beside the trainer: the trainer is the subject of
`pretrain.py`, and a file with two subjects is a file whose next reader has to guess which one an edit
belongs to. Here the subject is the CHOICE — add a target family by adding a builder and a table row,
and the parser's accepted values follow from the same table.
"""

from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass

from kneemri.models.encoder import Encoder
from kneemri.training.latent_mim import LatentMim, LatentMimConfig
from kneemri.training.masked_objective import MaskConfig, MaskedObjective
from kneemri.training.spectral_mae import SpectralMae, SpectralMaskConfig


@dataclass(frozen=True)
class Objectives:
    """What a run can be scored against. The masking, the corpus and the schedule are identical across
    them, so `--objective` is the one variable — which is what makes a pair of arms an A/B rather than
    two runs.

    The run's arguments and encoder are the instance because every builder is built against exactly
    those two and nothing else; `choices` needs neither, so it stays on the class.

    Neither builder is unit-tested: constructing one downloads a published checkpoint, which is exactly
    what `hygiene.small` forbids a unit test to reach for. What IS tested is the dispatch, `named`.
    """

    arguments: Namespace
    encoder: Encoder

    def spectral(self) -> MaskedObjective[MaskConfig]:  # devtools-ignore: test-mirror
        """The pixel objective: reconstruct the hidden patches, scored again in the frequency domain."""
        from transformers import AutoModel  # noqa: PLC0415 - a 90 MB import the unit gate must not pay for

        config = SpectralMaskConfig(
            mask_ratio=self.arguments.mask_ratio,
            spectral_weight=self.arguments.spectral_weight,
            highpass=self.arguments.highpass,
            patch=self.encoder.patch,
        )
        return SpectralMae(
            AutoModel.from_pretrained(self.encoder.repository), self.encoder.dim, config, self.encoder.normalization
        )

    def latent(self) -> MaskedObjective[MaskConfig]:  # devtools-ignore: test-mirror
        """The feature objective: predict an EMA teacher's normalised features at the hidden patches."""
        from transformers import AutoModel  # noqa: PLC0415 - a 90 MB import the unit gate must not pay for

        config = LatentMimConfig(
            mask_ratio=self.arguments.mask_ratio,
            patch=self.encoder.patch,
            retention=self.arguments.retention,
        )
        return LatentMim(AutoModel.from_pretrained(self.encoder.repository), config, self.encoder.normalization)

    @classmethod
    def choices(cls) -> tuple[str, ...]:
        """The names `--objective` accepts, so the parser and the dispatch cannot disagree."""
        return tuple(sorted(cls._table()))

    def named(self, name: str) -> MaskedObjective[MaskConfig]:
        """The objective a run asked for, built against its own arguments."""
        return self._table()[name](self)

    @staticmethod
    def _table() -> "dict[str, Callable[[Objectives], MaskedObjective[MaskConfig]]]":
        return {"spectral": Objectives.spectral, "latent": Objectives.latent}
