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

from kneemri.models.backbone import Encoder
from kneemri.training.latent_mim import LatentMim, LatentMimConfig
from kneemri.training.masked_objective import MaskConfig, MaskedObjective
from kneemri.training.spectral_mae import SpectralMae, SpectralMaskConfig


class Objectives:
    """What a run can be scored against. The masking, the corpus and the schedule are identical across
    them, so `--objective` is the one variable — which is what makes a pair of arms an A/B rather than
    two runs.

    Neither builder is unit-tested: constructing one downloads a published checkpoint, which is exactly
    what `hygiene.small` forbids a unit test to reach for. What IS tested is the dispatch, `named`.
    """

    @staticmethod
    def spectral(arguments: Namespace, encoder: Encoder) -> MaskedObjective[MaskConfig]:  # devtools-ignore: test-mirror
        """The pixel objective: reconstruct the hidden patches, scored again in the frequency domain."""
        from transformers import AutoModel  # noqa: PLC0415 - a 90 MB import the unit gate must not pay for

        config = SpectralMaskConfig(
            mask_ratio=arguments.mask_ratio,
            spectral_weight=arguments.spectral_weight,
            highpass=arguments.highpass,
            patch=encoder.patch,
        )
        return SpectralMae(AutoModel.from_pretrained(encoder.repository), encoder.dim, config, encoder.normalization)

    @staticmethod
    def latent(arguments: Namespace, encoder: Encoder) -> MaskedObjective[MaskConfig]:  # devtools-ignore: test-mirror
        """The feature objective: predict an EMA teacher's normalised features at the hidden patches."""
        from transformers import AutoModel  # noqa: PLC0415 - a 90 MB import the unit gate must not pay for

        config = LatentMimConfig(
            mask_ratio=arguments.mask_ratio,
            patch=encoder.patch,
            retention=arguments.retention,
        )
        return LatentMim(AutoModel.from_pretrained(encoder.repository), config, encoder.normalization)

    @classmethod
    def choices(cls) -> tuple[str, ...]:
        """The names `--objective` accepts, so the parser and the dispatch cannot disagree."""
        return tuple(sorted(cls._table()))

    @classmethod
    def named(cls, name: str, arguments: Namespace, encoder: Encoder) -> MaskedObjective[MaskConfig]:
        """The objective a run asked for, built against its own arguments."""
        return cls._table()[name](arguments, encoder)

    @classmethod
    def _table(cls) -> dict[str, Callable[[Namespace, Encoder], MaskedObjective[MaskConfig]]]:
        return {"spectral": cls.spectral, "latent": cls.latent}
