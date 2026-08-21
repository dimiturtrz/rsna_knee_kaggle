"""Which DINOv2 an encoder means, and how wide its features are.

`facebook/dinov2-small` with `DIM = 384` was written into the trainer, into the member-curve analysis
and into the Kaggle kernel as three independent constants, and the checkpoint recorded the variant as a
bare string that nothing read back. Changing encoder therefore meant changing a number in three files
and hoping they agreed — and a mismatched width does not fail loudly, it fails as a shape error deep in
a load, or not at all if the head happens to accept it.

The variant is the only free choice; the repository and the feature width follow from it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

SMALL = "small"
BASE = "base"
LARGE = "large"
V3_SMALL = "v3small"
V3_SMALL_PLUS = "v3splus"
V3_BASE = "v3base"
CONVNEXT_BASE = "cnxbase"

DINOV2 = "dinov2"
DINOV3 = "dinov3"
CONVNEXT = "convnext"


@dataclass(frozen=True)
class TokenConfig:
    """The two fields the head reads off a backbone's config, for a trunk whose config lacks them.

    `ConvNextV2Config` carries `hidden_sizes` — one width per stage — and no `hidden_size`, and it has
    no notion of register tokens at all. The head derives its feature width and its prefix count from
    whatever the loaded weights say, deliberately, because the weights are the only authority on what
    they emit. So a trunk that does not answer those questions is given something that does, rather
    than the head being taught a second way to ask.
    """

    hidden_size: int
    num_register_tokens: int = 0


@dataclass(frozen=True)
class TokenOutput:
    """What a transformers backbone returns, for a trunk that returns a feature map instead."""

    last_hidden_state: Float[Tensor, "n t d"]


class MapAsTokens(nn.Module):
    """A convolutional trunk wearing the vision transformer's output contract.

    A ConvNeXt emits `last_hidden_state` of shape (n, dim, h, w) — a feature MAP, where a ViT emits
    (n, 1 + patches, dim) — a token SEQUENCE with a class token in front. The head consumes the second
    shape, and teaching it the first would put a family test in the one place this project has kept
    free of them: the head is structural on purpose, so any module callable as `backbone(pixel_values=)`
    can be substituted as a constructor argument.

    So the map is flattened in row-major order — which is the same ordering a ViT's patch tokens
    already carry — and the map's global mean is prepended as a pseudo class token. That prefix is not
    a placeholder: mean-over-space is what a CNN classifier head pools before its linear layer, so the
    slot exists and holds the quantity a class token holds. Keeping it means `prefix_tokens` stays 1
    for every family and the head's `tokens[:, 1:]` still selects exactly the spatial grid.
    """

    config: TokenConfig
    """Annotated at class level because `nn.Module.__getattr__` types every attribute it serves as
    `Tensor | Module`, so a reader of `backbone.config.hidden_size` would otherwise be reading a
    Tensor as far as the type checker is concerned."""

    def __init__(self, trunk: nn.Module, dim: int) -> None:
        super().__init__()
        self.trunk = trunk
        self.config = TokenConfig(dim)

    @override
    def forward(self, pixel_values: Float[Tensor, "n c h w"]) -> TokenOutput:
        feature_map = self.trunk(pixel_values=pixel_values).last_hidden_state
        spatial = feature_map.flatten(2).transpose(1, 2)
        return TokenOutput(torch.cat([spatial.mean(dim=1, keepdim=True), spatial], dim=1))


@dataclass(frozen=True)
class Stage:
    """A run of blocks at one stride and one width — the unit an activation bill is summed over.

    A transformer has exactly one: every block sees the same token count at the same width, which is
    why a memory model could get away with reading the OUTPUT grid and multiplying by the width. A
    hierarchical convolutional trunk has four, each halving the grid twice over and doubling the width,
    and its bill is dominated by the FIRST of them — 7,744 tokens at 128 wide against the last stage's
    121 at 1024. Reading only the output grid there understates the demand by a factor of fifty.
    """

    blocks: int
    stride: int
    width: int

    def units(self, size: int) -> float:
        """Tokens times width times blocks — activation bytes up to one constant, at this read size."""
        return self.blocks * (size / self.stride) ** 2 * self.width


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

    depth: int = 12
    """How many blocks the trunk runs, which is the factor a per-block activation bill is missing.

    12 for every ViT this project has fitted — small, base and their DINOv3 siblings are all 12 deep,
    which is why depth could sit inside `GB_PER_SLICE` as a constant without anyone noticing. A
    hierarchical trunk states `stages` instead and this is ignored."""

    stages: tuple[Stage, ...] = ()
    """The block structure, for a trunk whose blocks do not all see the same grid at the same width.

    Empty for a transformer, where `depth`, `patch` and `dim` already say it: one stage, `depth` blocks,
    stride `patch`, width `dim`. Stated only where that shorthand is false."""

    params_m: float = 22.0
    """Trunk parameters in millions — the quantity the OPTIMISER's memory is a function of.

    Activations scale with tokens and width; weights, gradients and the two Adam moments scale with
    parameter count, and the VRAM cap held that second term as the constant `RESIDENT_GB = 0.22`
    fitted on this 22M trunk. That constant refused `v3base_dense` outright with nine gigabytes of the
    card free (knee-bj90): a ViT-B is four times the parameters and its optimiser state is four times
    the bytes, and the cap could not see it.

    Stated per encoder rather than counted off the loaded weights for the same reason `stages` is: the
    cap has to be derived BEFORE anything is fetched, since refusing a run that will not fit is what it
    is for."""

    def activation_units(self, size: int) -> float:
        """Summed tokens-times-width over every block, which is activation bytes up to one constant.

        THIS IS THE QUANTITY A VRAM CAP IS ACTUALLY A FUNCTION OF, and the thing four dead arms had in
        common is that the cap was a function of something else. It read the final grid — right for a
        transformer by accident, because a transformer's every block sees that grid, and structurally
        wrong for anything hierarchical.

        It is exact at the point `GB_PER_SLICE` was fitted by construction (12 blocks, 336 px, patch 14,
        384 wide) and returns the same ratio as the old form for EVERY vision transformer, since they
        are all 12 deep — so no DINOv2 or DINOv3 arm derives a different cap than it did before. Only a
        trunk with `stages` moves, which is the entire repair.
        """
        return sum(stage.units(size) for stage in self.stages or (Stage(self.depth, self.patch, self.dim),))

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

    def load(self, source: str | Path | None = None) -> nn.Module:
        """The trunk, ready to be handed to the head, from `source` or from this encoder's repository.

        Eight call sites fetched their own weights with `AutoModel.from_pretrained(encoder.repository)`
        and passed the result straight to the head, which was exactly right while every encoder was a
        vision transformer and every one of them emitted the same shape. A convolutional trunk does
        not, and the difference has to live SOMEWHERE — either once here, or eight times at the sites,
        each of which would then have to know which families need wrapping. This is the seam that keeps
        adding a family from costing a branch per consumer.

        `source` is a mounted directory in a kernel and None everywhere else; a kernel has no internet,
        so it holds the same weights under a path rather than a repository name.
        """
        # Imported here rather than at module scope because a Kaggle kernel can only reach transformers
        # once the kit is on `sys.path`, and this module is imported before that happens.
        from transformers import AutoModel  # noqa: PLC0415 - see above

        trunk = AutoModel.from_pretrained(source or self.repository)
        if self.family != CONVNEXT:
            return trunk
        return MapAsTokens(trunk, self.dim)

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
        BASE,
        "facebook/dinov2-base",
        768,
        mount="metaresearch/dinov2/PyTorch/base/1",
        family="dinov2",
        leaf="base",
        params_m=86.6,
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
        LARGE,
        "facebook/dinov2-large",
        1024,
        mount="metaresearch/dinov2/PyTorch/large/1",
        family="dinov2",
        leaf="large",
        params_m=304.0,
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
        params_m=21.6,
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
        params_m=29.0,
    ),
    V3_BASE: Encoder(
        V3_BASE,
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        768,
        mount="dimiturnt/dinov3-hf/vitb16",
        patch=16,
        family="dinov3",
        leaf="vitb16",
        params_m=86.6,
    ),
    # THE FIRST NON-TRANSFORMER, added 2026-08-21 because the transformer axis closed. `base_dense`
    # (DINOv2-base among seven DINOv2-small members) took the board from 0.930 to 0.931; `v3base_dense`
    # (DINOv3-base, same recipe, a whole different pretraining corpus and teacher) was worth -0.0007 as
    # a sixth seat. A second ViT family does not buy a second diversity seat, so the next member has to
    # differ in INDUCTIVE BIAS rather than in weights: local convolution and a stagewise hierarchy make
    # different mistakes than global attention over a flat grid, and that is the quantity an ensemble
    # is paid for.
    #
    # ConvNeXtV2-base is chosen for capacity parity with the member that worked: 87.7M parameters
    # against DINOv2-base's 86M, so the comparison is about the bias and not about size. FCMAE
    # pretraining keeps it self-supervised like the DINOs, then IN-22k finetuning — which is itself a
    # difference worth having, since every incumbent is SSL-only.
    #
    # STRIDE 32, NOT PATCH 32, and `patch` is the right field for it anyway: four stages halving twice
    # over, so the read must be a multiple of 32 for the grid to come out whole. 336 does NOT tile;
    # 352 gives an 11x11 grid of 121 tokens against a ViT's 576 at 336, which is why a convolutional
    # trunk of the same parameter count is cheaper per slice at a higher resolution.
    #
    # NO MOUNT, so this trains here and cannot be submitted yet. The weights are Apache-2.0 and mirror
    # cleanly the way `dinov3-hf` did; that is a dataset upload, not a code change, and it is deferred
    # until a member proves it is worth a seat.
    CONVNEXT_BASE: Encoder(
        CONVNEXT_BASE,
        "facebook/convnextv2-base-22k-224",
        1024,
        patch=32,
        family=CONVNEXT,
        leaf="convnextv2-base",
        params_m=87.7,
        # `depths` and `hidden_sizes` off ConvNeXtV2-base's own config, at the strides the four stages
        # run at. Stated rather than read from the loaded config because the cap has to be derived
        # BEFORE any weights are fetched — the whole point of it is to refuse a run that will not fit.
        stages=(Stage(3, 4, 128), Stage(3, 8, 256), Stage(27, 16, 512), Stage(3, 32, 1024)),
    ),
}
