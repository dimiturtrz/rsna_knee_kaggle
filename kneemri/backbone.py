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
LARGE = "large"


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
    # 86M parameters against 22M, and 768-wide features. The strongest public notebooks here run base.
    #
    # THE +0.0085 THIS COMMENT USED TO CLAIM WAS MEASURED ON A RULER WE NO LONGER USE, and on the
    # current one base does NOT clear the noise floor: rank24_s2026 (small) 0.8272 against
    # rankbase24_s2026 (base) 0.8283 is +0.0011, against a floor of 0.0053 that is the holdout's own
    # sampling error (knee-2rx). The old number predates the augmentation fix and the ledger rebuild
    # (knee-ghl), so it is not a second opinion about base — it is a measurement of a different setup.
    #
    # Left here rather than deleted because the stale claim nearly bought a 90-minute arm: it reads as
    # evidence FOR capacity, and our own current numbers say capacity is not the constraint (knee-7u5).
    BASE: Encoder(BASE, "facebook/dinov2-base", 768),
    # 300M parameters, 1024-wide. The rung ABOVE the public frontier: every strong notebook here stops
    # at base, and the reason is the scored kernel's T4 rather than anything about the task. That
    # constraint binds INFERENCE only — training happens on our own card, and the weights are mounted
    # from `metaresearch/dinov2/PyTorch/large/1`, which exists and is 1.2 GB.
    #
    # THAT ENDORSEMENT NO LONGER HOLDS EITHER, and it rested on the same stale ruler as base's. On the
    # current one: small 0.8272, base 0.8283, large_deep_rank24 0.8304 — and the large row also varies
    # depth, so its +0.0032 is not even a clean capacity measurement. All of it sits inside a 0.0053
    # floor. The lever that HAS cleared the floor since is whole-stack reading (+0.0147), which is a
    # data question, not a capacity one.
    LARGE: Encoder(LARGE, "facebook/dinov2-large", 1024),
}
