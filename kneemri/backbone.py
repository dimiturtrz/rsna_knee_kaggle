"""Which DINOv2 an encoder means, and how wide its features are.

`facebook/dinov2-small` with `DIM = 384` was written into the trainer, into the member-curve analysis
and into the Kaggle kernel as three independent constants, and the checkpoint recorded the variant as a
bare string that nothing read back. Changing encoder therefore meant changing a number in three files
and hoping they agreed — and a mismatched width does not fail loudly, it fails as a shape error deep in
a load, or not at all if the head happens to accept it.

The variant is the only free choice; the repository and the feature width follow from it.
"""

from dataclasses import dataclass

SMALL = "small"
BASE = "base"


@dataclass(frozen=True)
class Encoder:
    """One DINOv2 size: where to fetch it, and the width of the features it returns."""

    variant: str
    repository: str
    dim: int

    @property
    def mounted_name(self) -> str:
        """The Kaggle model slug this variant is mounted under in a kernel.

        Every public notebook for this competition mounts `metaresearch/dinov2/PyTorch/<variant>/1`,
        because the kernel has no internet and cannot fetch from the hub.
        """
        return f"metaresearch/dinov2/PyTorch/{self.variant}/1"


ENCODERS: dict[str, Encoder] = {
    SMALL: Encoder(SMALL, "facebook/dinov2-small", 384),
    # 86M parameters against 22M, and 768-wide features. The strongest public notebooks here run base;
    # ours has only ever run small, which makes this the largest single untested knob we have.
    BASE: Encoder(BASE, "facebook/dinov2-base", 768),
}
