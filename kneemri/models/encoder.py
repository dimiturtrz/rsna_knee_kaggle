"""What a registry entry IS: where a trunk comes from, what it costs, and the contract it wears.

Split from `backbone.py` when that file passed the god-file ceiling, along the seam the two halves
already had: this module is the VOCABULARY an encoder entry is written in, and `backbone` is the
DECLARATION of every trunk written in it. The direction is one-way — an entry cannot know the table it
sits in — which is what makes the seam a seam rather than a file-size compromise.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from torch import nn

DATASET_MOUNT = "dataset"
MODEL_MOUNT = "model"
"""Which Kaggle THING publishes a mount, because the two are read through different APIs and one of them
answers 403 to the other's endpoint. A kernel does not care — it mounts a path either way — but anything
asking Kaggle what a mount actually publishes has to address the right endpoint, and only the registry
that declared the mount knows which. `metaresearch/dinov2/PyTorch/small/1` is a model version;
`dimiturnt/dinov3-hf/vits16` is a directory inside a private dataset."""


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

    mount_kind: str = DATASET_MOUNT
    """Whether `mount` names a Kaggle DATASET or a Kaggle MODEL version — see the constants.

    Declared beside the mount rather than derived from its shape. A model slug happens to carry five
    segments today and a dataset mount two or three, and counting them would be a rule about Kaggle's
    current URL grammar dressed up as a fact about our weights. The registry that types the mount is the
    only place that knows what it typed."""

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

    normalization_evidence: str = ""
    """Where `normalization` came from, for a checkpoint that publishes no preprocessing document.

    Empty for every trunk shipped as a HuggingFace model directory: those state their own constants in a
    `preprocessor_config.json`, and `tests/integration/registry_to_checkpoints` holds the entry to that
    file rather than to anything written here. A bare `OrderedDict` release states nothing, so the
    integration test cannot compare — and a missing document reading as agreement is the exact hole that
    test was written to close.

    So the escape is EXPLICIT and it is VERIFIED: the test still asks HuggingFace for the document, and
    accepts this field only after confirming the repository really ships none. What it cannot verify is
    the argument, which is why the field is prose and not a flag — a bare `True` would record that
    somebody decided, not what they decided on."""

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

    builder: Callable[[str | Path | None], nn.Module] | None = None
    """How to build this trunk when it is NOT a HuggingFace model directory, or None when it is.

    `AutoModel.from_pretrained` needs a `config.json` beside the weights to name the class that rebuilds
    them. Every trunk here shipped that way until RadImageNet, which publishes a bare `OrderedDict` and
    so cannot say what it is. The architecture has to be stated instead (`StateDictWeights`), and the
    entry states it — for the reason `adapter` is held here rather than decided inside `load`: a branch
    keyed on "is this one of the odd ones" grows an arm per trunk in the one method whose whole purpose
    is that adding a family costs no branch per consumer."""

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
        trunk = self.builder(source) if self.builder is not None else self.from_hub(source)
        return trunk if self.adapter is None else self.adapter(trunk, self.dim)

    def from_hub(self, source: str | Path | None) -> nn.Module:  # devtools-ignore: test-mirror
        """The trunk as a HuggingFace model directory names it — the case every encoder but one is.

        No mirror test: exercising it means fetching real weights, which is a network call and the `small`
        gate forbids one in a unit test. It is covered where it was always covered — the integration suite
        `registry_to_checkpoints`, which loads every entry against its own snapshot.
        """
        # Imported here rather than at module scope because a Kaggle kernel can only reach transformers
        # once the kit is on `sys.path`, and this module is imported before that happens.
        from transformers import AutoModel  # noqa: PLC0415 - see above

        return AutoModel.from_pretrained(source or self.repository)

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

    def claims(self, mount: str) -> bool:
        """Whether a mount string names THIS encoder's weights, exactly or as a directory beneath them.

        The per-encoder half of the mount-kind question, kept here rather than in the registry scan that
        calls it because it is a fact about one entry and is testable without a table. A prefix match
        needs the separator: `dimiturnt/dinov3-hf/vits16` must not claim `dimiturnt/dinov3-hf/vits16plus`,
        which is the same substring trap `locates` was written for one layer down.
        """
        return bool(self.mount) and (mount == self.mount or mount.startswith(f"{self.mount}/"))
