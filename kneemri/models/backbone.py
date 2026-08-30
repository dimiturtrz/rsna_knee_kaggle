"""Which DINOv2 an encoder means, and how wide its features are.

`facebook/dinov2-small` with `DIM = 384` was written into the trainer, into the member-curve analysis
and into the Kaggle kernel as three independent constants, and the checkpoint recorded the variant as a
bare string that nothing read back. Changing encoder therefore meant changing a number in three files
and hoping they agreed — and a mismatched width does not fail loudly, it fails as a shape error deep in
a load, or not at all if the head happens to accept it.

The variant is the only free choice; the repository and the feature width follow from it.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from torch import nn

from kneemri.models.trunk_tokens import TrunkAsTokens

SMALL = "small"
BASE = "base"
LARGE = "large"
V3_SMALL = "v3small"
V3_SMALL_PLUS = "v3splus"
V3_BASE = "v3base"
CONVNEXT_BASE = "cnxbase"
VJEPA2_LARGE = "vjepa2l"
RAD_DINO = "raddino"
SWIN_BASE = "swinbase"
VIT_IN21K = "vitin21k"

DINOV2 = "dinov2"
DINOV3 = "dinov3"
CONVNEXT = "convnext"
VJEPA2 = "vjepa2"
RADDINO = "rad-dino"
SWIN = "swin"
VIT = "vit"


@dataclass(frozen=True)
class Normalization:
    """The mean and standard deviation a checkpoint was pretrained under — ITS contract, not ours.

    A PRETRAINED TRUNK CARRIES ITS OWN PREPROCESSING CONTRACT, and feeding it somebody else's is a
    handicap that raises no error. The model registered the ImageNet constants as buffers for every
    encoder, which was right six times out of seven by coincidence: DINOv2 (all three), DINOv3 (all
    three) and ConvNeXtV2-base all declare exactly those numbers in their own `preprocessor_config.json`.
    RAD-DINO does not — it was pretrained on greyscale radiographs at 0.5307 / 0.2583 — so it was
    receiving inputs offset 0.20 std from its pretraining distribution with a 12.8% scale error on top.

    That is the shape of the failure worth naming: the arm still trains, still logs a plausible curve,
    and still produces a real number supporting a false conclusion about the exact hypothesis it was
    built to test. Six encoders agreeing is what kept it invisible.

    DECLARED HERE rather than read from the checkpoint, for the reason `params_m` is: a Kaggle mount is
    not a HuggingFace snapshot. Ours holds `config.json` and the weights and no `preprocessor_config.json`
    at all, so a kernel reading the constants off the checkpoint would find nothing and fall back to the
    assumption this exists to remove. `assert_declared` is how the declaration is held to the checkpoint
    wherever that file IS present.
    """

    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @classmethod
    def declared(cls, preprocessor: Mapping[str, object]) -> "Normalization":
        """The constants a `preprocessor_config.json` states, for checking a declaration against it.

        The parameter is NOT called `config`, deliberately: `tests/integration/checkpoint_config_to_readers`
        scrapes every `config.get("key")` in the package and demands a fresh checkpoint stamp that key.
        A HuggingFace preprocessor file is a different document with a different vocabulary, and reading
        it through a receiver named `config` would tell that guard to look for `image_mean` in OUR stamp.
        """
        mean, std = preprocessor.get("image_mean"), preprocessor.get("image_std")
        if not isinstance(mean, Sequence) or not isinstance(std, Sequence):
            message = "preprocessor config carries no image_mean/image_std to read"
            raise ValueError(message)
        return cls(cls.triple(mean), cls.triple(std))

    @staticmethod
    def triple(values: Sequence[object]) -> tuple[float, float, float]:
        """The three channel constants, from a config that may state one per channel or one for all."""
        read = [float(value) for value in cast(Sequence[float], values)]
        stated = read * 3 if len(read) == 1 else read
        return (stated[0], stated[1], stated[2])

    def assert_declared(self, preprocessor: Mapping[str, object], variant: str) -> None:
        """Refuse a registry entry the checkpoint's own config contradicts, naming both sides.

        Only callable where the snapshot ships that file — which is every HuggingFace repository and
        none of our Kaggle mounts, so this guards the declaration at its source rather than at its use.
        """
        stated = Normalization.declared(preprocessor)
        if stated == self:
            return
        message = f"{variant} is declared {self.mean}/{self.std} and its checkpoint says {stated.mean}/{stated.std}"
        raise ValueError(message)


IMAGENET = Normalization((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
"""What every natural-image trunk in this registry was pretrained under, and the reason a per-encoder
field looked unnecessary for as long as the registry held only natural-image trunks."""

HALF = Normalization((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
"""The AugReg convention: centre on 0.5 and scale by 0.5, so a byte image lands on [-1, 1].

The SECOND trunk here that does not declare ImageNet, and it is worth naming that it is the second
rather than folding it into RAD-DINO's comment. RAD-DINO's constants are a property of its CORPUS —
greyscale radiographs really do have a different mean. These are a property of a pretraining CODEBASE:
every AugReg-trained ViT states them regardless of what it was trained on, because that pipeline never
adopted the ImageNet statistics. So a trunk's normalization cannot be predicted from its data any more
than from its architecture, which is the whole argument for the field existing."""

PREPROCESSOR_FILES = ("preprocessor_config.json", "video_preprocessor_config.json")
"""The documents a HuggingFace snapshot may state `image_mean`/`image_std` in, in the order to try.

A VIDEO trunk states the same two constants under a different filename: V-JEPA2 ships only
`video_preprocessor_config.json` and 404s on the other. A checker that knows one name therefore reads
the one encoder here whose pretraining corpus is not images as having no declaration AT ALL — which is
the quiet half of the failure `Normalization` exists to make loud, since an unread declaration and an
agreeing one are indistinguishable from the outside."""


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

    tubelet: int = 1
    """How many SLICES one token spans, for a trunk that tokenises the depth axis as well as the plane.

    1 for every trunk this project has fitted: a 2D trunk runs once per slice, so its token grid is a
    per-slice quantity and the slice loop in `DeviceBudget.needed` supplies the depth factor. A tubelet
    transformer tokenises `tubelet` slices at once, so the SAME window costs grid^2 / tubelet per slice —
    the cap is a per-slice function and this is the only place the depth axis enters it.

    THE DIRECTION MATTERS AND THE FIRST READING OF IT WAS BACKWARDS (knee-2nxh). A tubelet trunk emits
    grid^2 * slices/tubelet tokens for the whole window, which reads as MORE than a 2D trunk's grid^2
    only if the cap is per-STEP. It is not; it is per-slice, so the tubelet trunk costs LESS and an
    uncorrected cap OVERSTATES it. That is knee-bj90's failure mode, not an OOM: the run parks in
    `await_headroom` waiting on memory it would never have allocated, and never starts.
    """

    def per_block(self, size: int) -> float:
        """ONE block's tokens times width, PER SLICE — activation bytes up to one constant.

        Split out from `units` because the cost model asks two different questions of a trunk: what
        EVERY block costs to run forward, and what the OPEN ones cost to retain until their backward.
        The second is a suffix of the block list, and a suffix cannot be taken from a per-stage total.
        """
        return (size / self.stride) ** 2 * self.width / self.tubelet

    def units(self, size: int) -> float:
        """Tokens times width times blocks, PER SLICE — activation bytes up to one constant."""
        return self.blocks * self.per_block(size)


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

    normalization: Normalization = IMAGENET
    """The constants THIS checkpoint was pretrained under — see `Normalization` for what a wrong one costs.

    Defaulting to ImageNet is honest rather than lazy: it is what every natural-image trunk here really
    declares, and `tests/integration/registry_to_checkpoints` holds all nine entries to their own
    snapshots rather than to this sentence. A trunk pretrained on something else states its own, and the
    only one so far is RAD-DINO — so the fix that introduced this field invalidates exactly ONE arm and
    nothing already on disk."""

    depth: int = 12
    """How many blocks the trunk runs, which is the factor a per-block activation bill is missing.

    12 for every ViT this project has fitted — small, base and their DINOv3 siblings are all 12 deep,
    which is why depth could sit inside `GB_PER_SLICE` as a constant without anyone noticing. A
    hierarchical trunk states `stages` instead and this is ignored."""

    stages: tuple[Stage, ...] = ()
    """The block structure, for a trunk whose blocks do not all see the same grid at the same width.

    Empty for a transformer, where `depth`, `patch` and `dim` already say it: one stage, `depth` blocks,
    stride `patch`, width `dim`. Stated only where that shorthand is false."""

    tubelet: int = 1
    """How many slices one token spans — see `Stage.tubelet`, which is where it does its work.

    Carried here as well because a tubelet transformer states no `stages`: `depth`, `patch` and `dim`
    still describe it, and the depth-axis stride is the one thing that shorthand cannot say."""

    adapter: Callable[[nn.Module, int], nn.Module] | None = None
    """What to wrap the loaded trunk in so it speaks the head's output contract, or None if it already does.

    Held as a strategy rather than decided by a family test inside `load`, because the family test was
    `if self.family != CONVNEXT` and it was correct exactly until a second non-conforming family arrived.
    Swin is that second one and it does NOT want the same adapter — it emits a sequence already and is
    missing a prefix token, where ConvNeXt emits a map and is missing both. A branch keyed on family
    would have had to grow one arm per trunk in the one method whose docstring says it exists so that
    adding a family costs no branch per consumer; this way the entry states its own answer."""

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

    @property
    def blocks(self) -> tuple[Stage, ...]:
        """One entry per block of the trunk, in run order — the stage each block belongs to.

        Held as blocks rather than as stages because the two things a cost model asks are a SUM over
        all of them and a SUM OVER A SUFFIX, and the suffix — what `unfreeze_last` opened — crosses
        stage boundaries. A ConvNeXt's last six blocks are three of stage four and three of stage
        three, at two different widths on two different grids.

        THIS IS ALSO THE TRUNK'S REAL DEPTH, which `depth` is not for a hierarchical trunk: `cnxbase`
        states `depth` 12 (the field's default, meaningless once `stages` is given) and runs 36. Every
        clamp of `unfreeze_last` therefore belongs here rather than at a caller reading `depth`.
        """
        stages = self.stages or (Stage(self.depth, self.patch, self.dim, tubelet=self.tubelet),)
        return tuple(stage for stage in stages for _ in range(stage.blocks))

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
        return sum(block.per_block(size) for block in self.blocks)

    def retained_units(self, size: int, opened: int) -> float:
        """The same quantity over the OPEN blocks only — what must survive to a backward.

        `activation_units` is what the trunk costs to run; this is what it costs to remember, and the
        two scale differently because a frozen block's activations are freed the moment the next block
        consumes them. The cost model weighs them separately (`DeviceBudget.needed`), and until
        2026-08-25 it did not: it multiplied the whole-trunk sum by a scalar in the block COUNT, which
        is only right when every block costs the same. On a ViT it is. On the two trunks that are not
        ViTs it prices the open suffix at the trunk's AVERAGE block, and their suffixes are nowhere
        near average — `cnxbase`'s last six blocks are 9.7% of its bill, where six of twelve uniform
        blocks would be 50%.

        `opened` is clamped HERE, against the real block count, because that is the only place that
        knows it — a caller clamping against `depth` gives a fully-open ConvNeXt twelve blocks of
        retention out of the thirty-six it actually opens.
        """
        blocks = self.blocks
        held = blocks[len(blocks) - min(max(opened, 0), len(blocks)) :] if opened > 0 else ()
        return sum(block.per_block(size) for block in held)

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
        return trunk if self.adapter is None else self.adapter(trunk, self.dim)

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
        SMALL, "facebook/dinov2-small", 384, mount="metaresearch/dinov2/PyTorch/small/1", family=DINOV2, leaf="small"
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
        family=DINOV2,
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
        family=DINOV2,
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
        family=DINOV3,
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
        family=DINOV3,
        leaf="vits16plus",
        params_m=29.0,
    ),
    V3_BASE: Encoder(
        V3_BASE,
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        768,
        mount="dimiturnt/dinov3-hf/vitb16",
        patch=16,
        family=DINOV3,
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
        adapter=TrunkAsTokens.from_map,
        # `depths` and `hidden_sizes` off ConvNeXtV2-base's own config, at the strides the four stages
        # run at. Stated rather than read from the loaded config because the cap has to be derived
        # BEFORE any weights are fetched — the whole point of it is to refuse a run that will not fit.
        stages=(Stage(3, 4, 128), Stage(3, 8, 256), Stage(27, 16, 512), Stage(3, 32, 1024)),
    ),
    # THE FIRST TRUNK THAT WAS NOT PRETRAINED WITHOUT LABELS, added 2026-08-30. Every encoder above is
    # self-supervised — DINOv2 and DINOv3 by distillation, ConvNeXtV2 by masked autoencoding, V-JEPA2 by
    # latent prediction — so "a different trunk" has so far always meant a different way of having no
    # labels. Swin-base is IN-22k SUPERVISED: 21,841 human-assigned classes, an objective that rewards
    # exactly the discriminative structure our head is then asked to find, and the one axis of pretraining
    # this project has never once varied. That is the diversity claim, and it is about the OBJECTIVE
    # rather than about the architecture, which is why it is worth a run after ConvNeXtV2's read PARTIAL.
    #
    # THE ARCHITECTURE DIFFERS TOO, and differently from ConvNeXt's. Shifted-window attention is neither
    # global attention over a flat grid (DINOv2) nor pure local convolution (ConvNeXtV2): attention within
    # a 12x12 window, with the windows displaced every other block so information crosses their seams.
    # `knee-y59c` measured distance-from-DINOv2 anti-correlating with solo strength across seven trunks,
    # so this sits deliberately in the MIDDLE of that axis rather than at the far end where `mae_matched`
    # (0.6735) and `focus_mcl` (0.5332) died.
    #
    # 384 IS THE ONLY LEGAL READ, and `require_tiling` is NOT sufficient to enforce it. The window size is
    # baked into the pretrained relative-position tables at 12, and a 12x12 window tiles a 384 px read at
    # stride 32 exactly. 352 tiles the stride and NOT the window: transformers will pad and interpolate
    # rather than refuse, handing back a plausible answer computed on a geometry the tables never saw.
    # The arm therefore pins --train-size 384 and this comment is the reason it may not be swept.
    #
    # NO MOUNT YET, so it trains here and cannot be submitted until the weights are mirrored. Apache-2.0,
    # so mirroring is permitted outright — no licence question of the kind DINOv3 needed — and it is the
    # same dataset upload `dinov3-hf` already is. Deferred until a member earns the seat.
    SWIN_BASE: Encoder(
        SWIN_BASE,
        "microsoft/swin-base-patch4-window12-384-in22k",
        1024,
        patch=32,
        family=SWIN,
        # THE DATASET'S SLUG, not a prettier name for the trunk. `leaf` has exactly one consumer,
        # `locates`, which requires it as an exact path SEGMENT of the mounted directory — and a flat
        # mount's directory IS the slug, `/kaggle/input/swin-base-384`. Spelled `swin-base` this entry
        # matched no mount, so `find_encoder` raised and `kernel_run` dropped the member from the entry
        # with the other members still loading: a submission that scores, without the thing it was for.
        leaf="swin-base-384",
        params_m=86.9,
        adapter=TrunkAsTokens.from_sequence,
        # FLAT, like `dimiturnt/rad-dino` and unlike `dimiturnt/dinov3-hf/<leaf>`: the leaf shape exists
        # because DINOv3 ships four variants from one dataset, and there is one Swin. Written with
        # `SwinModel.save_pretrained`, so the mirror holds exactly what `load` reads back — verified by
        # loading it from disk through this entry, which is what a kernel does with a mounted dataset.
        # Apache-2.0 and ungated (read from the HF API, not the model card), so §2.6.a is met by the
        # upstream repository and the mirror exists only because a kernel has no internet.
        mount="dimiturnt/swin-base-384",
        # `depths` and `embed_dim` off Swin-base's own config, at the strides its four stages run at —
        # stated ahead of any fetch for the same reason ConvNeXtV2's are: the cap refuses a run that
        # will not fit, so it cannot wait for the weights to say.
        stages=(Stage(2, 4, 128), Stage(2, 8, 256), Stage(18, 16, 512), Stage(2, 32, 1024)),
    ),
    # THE OBJECTIVE AXIS WITH THE ARCHITECTURE HELD FIXED, added 2026-08-30. `swinbase` above is this
    # registry's first supervised trunk and it varies TWO things at once against every incumbent: the
    # pretraining objective AND the architecture. Whatever it reads, it cannot say which half carried it.
    # This entry can. It is an exact architectural twin of `v3base` — ViT-B/16, twelve blocks, 768 wide,
    # global attention over the same patch grid — and differs from it in the objective ALONE: AugReg
    # supervised IN-21k against DINOv3 self-distillation on LVD-1689M. `v3base_dense` already ran the
    # dense recipe on the twin and read -0.0007 as a sixth seat, so the control is on disk.
    #
    # `knee-y59c` MEASURED SOLO STRENGTH ANTI-CORRELATING WITH DISTANCE FROM DINOv2 across seven trunks,
    # and an architectural twin sits at ~0 on that axis — which is why this is expected to produce a real
    # number rather than another `mae_matched` (0.6735) or `focus_mcl` (0.5332). A strong solo with a
    # worthless seat would itself be the finding: it would say the objective moves the trunk's COMPETENCE
    # and not its OPINION, which is the quantity `knee-m69u` asks about and the one Swin confounds.
    #
    # NOT IMAGENET, and this one is not a corpus story. See `HALF`.
    #
    # 224 IS THE PRETRAINED READ and 16 tiles it at 14x14. Larger reads interpolate the position
    # embeddings, which every ViT here already does — so unlike Swin's window tables this may be swept.
    #
    # NO MOUNT YET. Apache-2.0 and ungated, so mirroring is permitted outright and is the same dataset
    # upload `dimiturnt/swin-base-384` is; deferred until a member earns the seat.
    VIT_IN21K: Encoder(
        VIT_IN21K,
        "google/vit-base-patch16-224-in21k",
        768,
        patch=16,
        family=VIT,
        normalization=HALF,
        leaf="vit-base-in21k",
        params_m=86.4,
    ),
    # THE FIRST TRUNK THAT WAS NOT PRETRAINED ON FLAT IMAGES, added 2026-08-21 after `through_dense`
    # answered the cheap version of this question and answered it flat. That arm put slices i-1, i, i+1
    # into the three input channels and read -0.0020 against its control, and the mechanism its null
    # leaves standing is precise: `Conv2d(3, D, k=14, s=14)` over three slices is IDENTICAL to
    # `Conv3d(1, D, k=(3,14,14), s=(1,14,14))`, so a 2D stem at window 3 collapses the window to ONE
    # linear projection per output dimension and emits ONE depth position. What it cannot do at any
    # window size is emit SEVERAL. That is the residue, and a tubelet trunk is the only thing that buys
    # it — which makes this the expensive version of a question already asked cheaply, not a new guess.
    #
    # PICKED ON LICENCE, NOT ON BENCHMARKS. The HF API says MCG-NJU/videomae-base and
    # facebook/timesformer-base-finetuned-k400 are both cc-by-nc-4.0, while this is MIT. A
    # non-commercial clause plausibly propagates to weights trained from it, which is a competition
    # ELIGIBILITY question rather than a modelling one, and it is cheaper to answer with one API call
    # than to discover after a member scores.
    #
    # THE RISK, STATED RATHER THAN DISCOVERED. It is pretrained on natural VIDEO, so this is natural-
    # domain pretraining again — and our own cautionary instance is on the ledger, where DINOv3, a
    # stronger natural-image encoder, scored ~0.04 WORSE than DINOv2 with the deficit in focal findings.
    # Worse, video is genuinely TEMPORAL and MRI depth is SPATIAL; nothing measured anywhere here says a
    # trunk pretrained on motion transfers to a spatial third axis. That is exactly why the first read
    # is a frozen embedding pass plus a head fit rather than an arm.
    #
    # NO MOUNT, so this trains here and cannot be submitted until the weights are mirrored — a dataset
    # upload, deferred until the probe says it is worth one.
    VJEPA2_LARGE: Encoder(
        VJEPA2_LARGE,
        "facebook/vjepa2-vitl-fpc64-256",
        1024,
        patch=16,
        family=VJEPA2,
        leaf="vjepa2-vitl",
        # V-JEPA2 has no official Kaggle mirror, so the snapshot is published as a private dataset the
        # same way DINOv3 is, one directory per variant named for the leaf `locates` matches on. Without
        # this the variant trains locally and cannot ship: `package_members.shippable` refuses a member
        # whose encoder no dataset mirrors, which is the half `knee-rpkw`'s recorded shippability
        # clearance did not check — it verified the kit carried the CODE and concluded the member could
        # fire, while a kernel has no internet and reaches weights only through a mount.
        mount="dimiturnt/vjepa2-hf/vjepa2-vitl",
        depth=24,
        # The only tubelet trunk in the registry: `fpc64` tokenises TWO frames per token, so the same
        # window costs half the per-slice activations a 2D ViT-L would. Without this the cap OVERSTATES
        # it by 2x and parks the run in `await_headroom` on memory it would never allocate (knee-2nxh).
        tubelet=2,
        # Counted off the loaded weights once, here, rather than guessed from "ViT-L": 325.97M.
        params_m=326.0,
    ),
    # THE FIRST TRUNK NOT PRETRAINED ON NATURAL IMAGES, added 2026-08-23. Every encoder above it —
    # DINOv2, DINOv3, ConvNeXtV2, V-JEPA2 — was pretrained on photographs or on video of the world, and
    # the failure mode they share is stated on the ledger: `base_dense` reaches 0.8371 holdout at train
    # loss 0.8456 while `cnx_dense` reaches 0.7927 at a BETTER train loss of 0.8332. A trunk that fits
    # our pixels harder generalises 0.044 worse on them. That gap is what a domain-matched
    # initialisation is for, and it is the only mechanism left priced above the 1.5% bar (knee-ombx).
    #
    # AN EXACT CONTROL EXISTS, which is the whole reason this one is worth an arm. RAD-DINO is
    # FINETUNED FROM `facebook/dinov2-base` — same ViT-B/14, same 768 width, same 12 blocks, no
    # registers — so it is a drop-in for BASE through the identical `AutoModel` path, with no
    # conversion, no wrapper and no new dependency. `base_dense_s2026` is that encoder at this exact
    # recipe (0.8512). Swapping only the initialisation weights therefore reads the domain question
    # with ONE variable, which no previous encoder arm here has managed: every other family change
    # moved architecture and pretraining together.
    #
    # THE DOMAIN MATCH IS PARTIAL AND IS NOT OVERSOLD. 882,775 CHEST RADIOGRAPHS (MIMIC-CXR, CheXpert,
    # NIH-CXR, PadChest, BRAX) — grayscale clinical anatomy with radiological tissue contrast, and
    # neither the acquisition physics nor the body part of a knee MRI. So it prices below the class:
    # +0.005-0.015 plausible against the mechanism's 1.5-3%.
    #
    # IT ALSO PRICES RadImageNet, which is the better-matched and far dearer version of the same
    # question — 1.35M radiology images across 11 anatomies, its own published evaluation including ACL
    # and meniscus tear detection on MR, and Keras `.h5` weights with no PyTorch port, so 2-4 h of
    # conversion carrying a silent-mismatch failure mode that would read exactly like a null. If
    # radiology initialisation moves nothing against an EXACT control here, that conversion is not
    # worth buying; this is the cheap half of a cost-tiered pair.
    #
    # LICENCE MIT, checked before the arm rather than after, the same way V-JEPA2's was. MOUNTED on
    # 2026-08-23, mid-arm and BEFORE the number: deferring the upload "until a seat is earned" had the
    # ordering backwards, because an unmirrored encoder makes a WINNING member offline-only, so the
    # deferred work was not the reward for a good number but the thing standing between one and a slot.
    #
    # A DATASET OF ITS OWN, FLAT, rather than a leaf under a shared one like `dinov3-hf`. The first
    # attempt copied that layout and lost it: `kaggle datasets create --dir-mode zip` zips a subdirectory
    # and Kaggle unpacks it into the ROOT, so `dimiturnt/raddino-hf` holds `config.json` at top level and
    # the `rad-dino/` segment `locates` matches on does not exist. That was caught by listing `dinov3-hf`
    # — which DOES report `vits16/...` prefixes — rather than by a kernel, which is where it was heading.
    # Naming the dataset for the encoder makes the mount path its own leaf and needs no directory at all.
    #
    # THE ONE ENCODER HERE THAT IS NOT NORMALISED LIKE THE OTHERS, and until 2026-08-25 it was fed the
    # ImageNet constants anyway, because the model registered them as buffers for every trunk alike. Its
    # own `preprocessor_config.json` says 0.5307 / 0.2583 — greyscale, one number repeated across the
    # three channels — so the trunk was seeing a 0.20-std offset with a 12.8% scale error. That makes
    # `raddino_dense`'s 0.8137 against `base_dense_s2026`'s 0.8512 a reading of the mismatch as much as
    # of the domain, on top of the run being TRUNCATED at epoch 6 of 12 (knee-8wm6). The exact-control
    # argument above is undamaged and unmeasured: swapping only the initialisation is still the clean
    # single-variable read, and it has not actually been taken yet.
    RAD_DINO: Encoder(
        RAD_DINO,
        "microsoft/rad-dino",
        768,
        mount="dimiturnt/rad-dino",
        family=RADDINO,
        leaf="rad-dino",
        params_m=86.6,
        normalization=Normalization((0.5307, 0.5307, 0.5307), (0.2583, 0.2583, 0.2583)),
    ),
}
