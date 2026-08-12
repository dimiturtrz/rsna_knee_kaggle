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
V3_SMALL = "v3small"
V3_BASE = "v3base"


@dataclass(frozen=True)
class Encoder:
    """One DINOv2 size: where to fetch it, and the width of the features it returns."""

    variant: str
    repository: str
    dim: int
    mount: str | None = None
    """Where a kernel finds these weights. None means there is no Kaggle mirror, so a member fitted on
    this encoder can be TRAINED here and cannot be submitted until one is arranged."""

    patch: int = 14
    """DINOv2 is patch 14 and DINOv3 is patch 16, which decides which read sizes tile cleanly."""

    tokens: tuple[str, ...] = ()
    """Lowercase fragments that ALL appear in this encoder's mounted path and in no sibling's.

    A kernel has no internet, so the weights arrive as a mount and the code has to recognise which
    directory holds which encoder. That recognition used to be `"dinov2" in path` plus a split on the
    literal string `dinov2` — which cannot see a DINOv3 at all, and cannot tell `vits16` from `vitb16`
    because those are not what our variants are called. Stating the fragments per encoder means a new
    encoder is a registry row rather than an edit inside the kernel.
    """

    def locates(self, path: str) -> bool:
        """Whether a mounted directory holds THIS encoder's weights.

        Matched rather than merely found: mounting small and base together and taking whichever sorted
        first builds a member at the wrong width, which surfaces as a shape error during load if we are
        lucky and as a silently wrong model if the head happens to accept it.
        """
        lowered = path.lower().replace("\\", "/")
        return bool(self.tokens) and all(token in lowered for token in self.tokens)

    def require_tiling(self, size: int) -> None:
        """Refuse a read size the patch grid cannot tile, naming the sizes that would work.

        A vision transformer cuts the image into `patch`-sized squares, so a side that is not a
        multiple of `patch` has no clean grid. DINOv2 is patch 14 and DINOv3 is patch 16, and they do
        NOT agree about which sizes are legal: 336 tiles under both, 392 tiles only under 14, 384 only
        under 16. Our best DINOv2 recipe reads 392, which a DINOv3 run cannot copy.

        Checked BEFORE a run allocates anything, because the alternative is discovering it after the
        weights are downloaded and the cache is mapped — or worse, not discovering it, since a
        transformers backbone may silently interpolate its position embeddings to fit and hand back a
        plausible answer computed on a grid the weights were never trained for.
        """
        if size % self.patch == 0:
            return
        below = (size // self.patch) * self.patch
        message = (
            f"{self.variant} is patch {self.patch}, which cannot tile a {size} px read — "
            f"use {below} or {below + self.patch}"
        )
        raise ValueError(message)

    @property
    def mounted_name(self) -> str:
        """The Kaggle model slug this variant is mounted under in a kernel.

        Every public notebook for this competition mounts `metaresearch/dinov2/PyTorch/<variant>/1`,
        because the kernel has no internet and cannot fetch from the hub. DINOv3 has NO official
        mirror there — only community re-uploads — so it is stated per encoder rather than derived,
        and asking for a mount that does not exist raises here instead of failing inside a kernel.
        """
        if self.mount is None:
            message = (
                f"{self.variant} has no Kaggle mount: it can be trained locally but not submitted "
                "until its weights are published as a dataset (a Meta-licence question, not a code one)"
            )
            raise ValueError(message)
        return self.mount


ENCODERS: dict[str, Encoder] = {
    SMALL: Encoder(
        SMALL, "facebook/dinov2-small", 384, mount="metaresearch/dinov2/PyTorch/small/1", tokens=("dinov2", "small")
    ),
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
    BASE: Encoder(
        BASE, "facebook/dinov2-base", 768, mount="metaresearch/dinov2/PyTorch/base/1", tokens=("dinov2", "base")
    ),
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
    LARGE: Encoder(
        LARGE, "facebook/dinov2-large", 1024, mount="metaresearch/dinov2/PyTorch/large/1", tokens=("dinov2", "large")
    ),
    # DINOv3, authorised 2026-08-12. Patch 16 rather than 14, so the read sizes that tile cleanly are
    # multiples of 16: 336 (441 tokens), 384 (576), 448 (784). Note 392 - our best DINOv2 read size -
    # does NOT tile at patch 16, and 384 costs exactly what DINOv2 costs at 336.
    #
    # It also carries FOUR register tokens where v2 has none, which is why the head derives its prefix
    # count from the backbone's config instead of assuming one class token: folding registers into the
    # patch mean would average in the high-norm outliers they exist to absorb.
    #
    # THE MOUNT IS OURS. Kaggle has no official metaresearch/dinov3 — the only DINOv3 there is Keras's,
    # in KerasHub preset format, which `AutoModel.from_pretrained` cannot read. So the transformers
    # snapshot is published as a PRIVATE dataset, `dimiturnt/dinov3-hf`, holding one directory per
    # variant. Private is deliberate: it is enough for our own kernel to mount, and it is not a
    # redistribution to the public.
    #
    # THE LICENCE PERMITS THIS, checked before uploading rather than after: the DINOv3 licence allows
    # making the materials available to a third party "only under the terms of this Agreement", with a
    # copy of the Agreement provided alongside. The HF snapshot ships LICENSE.md and it was uploaded
    # with the weights, so that condition is met by construction. Publication of results additionally
    # requires acknowledging DINOv3, which the README must carry if a v3 member ever scores.
    V3_SMALL: Encoder(
        V3_SMALL,
        "facebook/dinov3-vits16-pretrain-lvd1689m",
        384,
        mount="dimiturnt/dinov3-hf/vits16",
        patch=16,
        tokens=("dinov3", "vits16"),
    ),
    V3_BASE: Encoder(
        V3_BASE,
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        768,
        mount="dimiturnt/dinov3-hf/vitb16",
        patch=16,
        tokens=("dinov3", "vitb16"),
    ),
}
