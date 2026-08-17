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
V3_SMALL_PLUS = "v3splus"
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

    family: str = ""
    """The encoder line, as it appears somewhere in the mounted path: `dinov2` or `dinov3`."""

    leaf: str = ""
    """The directory named for THIS encoder, matched as a whole path segment and never as a substring.

    A kernel has no internet, so the weights arrive as a mount and the code has to recognise which
    directory holds which encoder. Substring matching was enough while the sizes were `small`, `base`
    and `large`, and stops being enough the moment DINOv3's ViT-S+ arrives: `vits16` is a substring of
    `vits16plus`, so a substring rule makes both encoders claim both mounts — the LOUD version of which
    is a kernel refusing to start, and the quiet version is a member built on the wrong weights at the
    same 384 width, which loads without a shape error and scores like nothing in particular.
    """

    def locates(self, path: str) -> bool:
        """Whether a mounted directory holds THIS encoder's weights.

        Matched rather than merely found: mounting two encoders together and taking whichever sorted
        first builds a member on weights it was not fitted on, which surfaces as a shape error during
        load if we are lucky and as a silently wrong model if the widths happen to agree — and between
        `vits16` and `vits16plus` they do agree, both being 384 wide.
        """
        segments = path.lower().replace("\\", "/").split("/")
        return bool(self.leaf) and self.family in path.lower() and self.leaf in segments

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
    def mounted_source(self) -> str:
        """The Kaggle source a kernel must attach for this variant — and NOT always a model slug.

        Two kinds live in one field on purpose, because a kernel needs one answer to "what must be
        attached". DINOv2 arrives as a Kaggle *model*, `metaresearch/dinov2/PyTorch/<variant>/1`, which
        every public notebook mounts because a kernel has no internet. DINOv3 has no official mirror
        there — only a KerasHub preset `AutoModel` cannot read — so ours arrives as a private *dataset*,
        `dimiturnt/dinov3-hf/<leaf>`, and a kernel declares it under `dataset_sources` instead. Which
        list it belongs in is the kernel metadata's business; that it must be declared at all is this.

        Asking for a variant with no mount raises here rather than inside a kernel, where the answer
        costs a submission slot.
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
        SMALL, "facebook/dinov2-small", 384, mount="metaresearch/dinov2/PyTorch/small/1", family="dinov2", leaf="small"
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
        BASE, "facebook/dinov2-base", 768, mount="metaresearch/dinov2/PyTorch/base/1", family="dinov2", leaf="base"
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
        LARGE, "facebook/dinov2-large", 1024, mount="metaresearch/dinov2/PyTorch/large/1", family="dinov2", leaf="large"
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
        family="dinov3",
        leaf="vits16",
    ),
    # ViT-S+/16: the SAME 384-wide, 12-layer, 6-head, patch-16 shape as vits16 with a wider MLP —
    # 28.7M parameters against 21.6M. A drop-in for the head, and the apples-to-apples partner for the
    # small comparison, since capacity is the only thing that differs.
    V3_SMALL_PLUS: Encoder(
        V3_SMALL_PLUS,
        "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        384,
        mount="dimiturnt/dinov3-hf/vits16plus",
        patch=16,
        family="dinov3",
        leaf="vits16plus",
    ),
    V3_BASE: Encoder(
        V3_BASE,
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        768,
        mount="dimiturnt/dinov3-hf/vitb16",
        patch=16,
        family="dinov3",
        leaf="vitb16",
    ),
}
