"""What one training run's command line SAYS, as a value: `kneemri.training.options`.

Separated from `train.py` (which runs a fit) and from `run_setup.py` (which builds what a fit needs)
because the three are different jobs and only this one is imported by things that never train —
`arm_check` parses a queued invocation with it, `switch_coverage` diffs its fields against the arms,
`member_record` stamps it into a checkpoint. Those consumers were importing the training entrypoint,
and with it `transformers`, to read a dataclass.
"""

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast, get_args, get_type_hints

from core.data.corpus.crop_centre import CROP_MM, DEFAULT_CENTRE, Crop
from core.data.corpus.folds import FitPositions
from core.data.corpus.slot_reader import IMAGE_SIZE, SLICE_BAND, SPREAD, Band, Reading
from core.data.corpus.slots import CANONICAL_SCHEME, FIRST_MAXIMAL, Routing
from core.data.corpus.study_cache import StudyCache
from core.data.labels.targets import CONFIDENCE_FLOOR, Supervision
from kneemri.models.backbone import SMALL
from kneemri.models.model import StudyDesign
from kneemri.models.slice_pooling import CHANNELS
from kneemri.models.slot_head import ATTENDED
from kneemri.models.token_pooling import CLS_MEAN
from kneemri.training.adaptation import Adaptation
from kneemri.training.flag_vocabulary import FlagVocabulary
from kneemri.training.gradient_clip import UNCLIPPED
from kneemri.training.mixup import OFF
from kneemri.training.pretrained_encoder import ALL_BLOCKS
from kneemri.training.sharpness import FLAT
from kneemri.training.trainer import ONE_CYCLE_WARMUP

NOT_STATED = "None"
"""How `recipe` spells a field that was never given — `str(None)`, since it stringifies every value.

It is ambiguous for the two `str | None` fields, where the literal text and the absent value have the
same spelling, and unambiguous for every other field: no int, float, Path or bool stringifies to this."""

_READERS: dict[type, Callable[[str], object]] = {
    int: int,
    float: float,
    str: str,
    Path: Path,
    bool: lambda text: text == str(True),
}
"""How each field type reads back off the string `recipe` wrote — `bool` being the one that cannot use
its own constructor, since `bool("False")` is True and would silently arm every switch a run left off."""


@dataclass(frozen=True)
class TrainingOptions:
    """Everything one run varies, parsed once.

    The parser lived in `main`, where it grew a statement per flag and pushed the function over the
    complexity gate twice in one session. A flag is configuration, not control flow, so it belongs in a
    type — and a type can then be constructed in a test without a command line.

    IT READS LCOM4=6 IN THE COHESION REPORT AND MUST. The fields are flat because `parse` builds this
    straight from argparse's dest names, so every knob has to land as a scalar; the grouping the rest of
    the code wants — `routing`, `crop`, `band`, `reading`, `cache`, `supervision`, `positions` — is
    therefore made of properties, each touching a disjoint few of those scalars. That is exactly the
    shape LCOM4 counts as six unrelated responsibilities, and splitting the class into six would break
    the two things this type exists for: one `**vars(parse_args())` constructor, and one `recipe` dict
    that is the whole account of what a run parsed. Advisory report, correctly ignored — don't re-audit.
    """

    fold: int = 0
    folds: int = 5
    exclude_fold: int = -1
    """A fold dropped from training as well as from the holdout; -1 excludes nothing. What makes a
    teacher fitted at `--fold k` genuinely out-of-fold for a student scored on fold 0."""
    epochs: int = 8
    batch: int = 12
    accumulate: int = 1
    """Micro-batches per optimiser step: the EFFECTIVE batch is `batch * accumulate`, at the activation
    memory of `batch` alone. 1 is the loop every arm in the ledger ran."""
    mixup: float = OFF
    """Beta concentration for blending each batch with a permutation of itself; 0 is off. The first
    LABEL-SPACE regulariser this project has — see `kneemri.training.mixup` for why the appearance-only
    one it already had left the overfitting untouched."""
    slot_dropout: float = 0.0
    """Probability a PRESENT slot is withheld from a training study; 0 is off and is what every member
    on disk was fitted under. The first SUPERVISION-axis regulariser this project has — mixup perturbs
    labels and the intensity block perturbs appearance, and neither changes which acquisitions a study
    is made of. See `kneemri.training.augmentation.AugmentConfig.slot_dropout`."""
    scale_jitter: float = 0.08
    """Half-width of the uniform field-of-view jitter; 0.08 is the public baseline's and was hardcoded
    until knee-su2sw, so it is what every member in the pool was fitted under. The first EXTENT-axis
    knob this project has: mixup perturbs the label, the intensity block perturbs appearance, slot
    dropout perturbs which acquisitions are present, and none of them changes how much of the knee is
    in frame. See `kneemri.training.augmentation.AugmentConfig.scale_jitter`."""
    padding_mode: str = "border"
    """What the augmentation resample reads outside the frame; "border" was hardcoded until knee-10f10
    and is what every ledger number was measured under. The companion of `scale_jitter` rather than an
    axis of its own — the rule only bites where the sampling grid leaves the cached crop, which is where
    a widened jitter spends its draws. See `kneemri.training.augmentation.AugmentConfig.padding_mode`."""
    slices: int = 12
    image_size: int = IMAGE_SIZE
    train_size: int = 0
    """0 means read the cache at its own resolution; anything else resamples to that size."""
    scheme: str = CANONICAL_SCHEME
    crop_mm: float = CROP_MM
    centre: str = DEFAULT_CENTRE
    band_low: float = SLICE_BAND[0]
    band_high: float = SLICE_BAND[1]
    spacing: str = SPREAD
    """How the slices sit inside the band. A scalar field rather than part of `band` because
    `Band.recorded` is a pair of floats every manifest on disk holds, and argparse's dest names land
    as fields here."""
    tie: str = FIRST_MAXIMAL
    seed: int = 2026
    split_seed: int = 2026
    train_fraction: float = 1.0
    """The share of the TRAINING studies actually fitted on; the holdout is never touched.

    The knob a data-scaling point needs, and it is separate from `--studies` for the reason a scaling
    point is not a smoke: `sampled` cuts BOTH halves so the loop still runs end to end on a few dozen
    studies, and a holdout that moves with the training size makes the two points incomparable. This
    cuts one side only, so the number it produces is scored on exactly the studies the ledger's
    full-data rows were scored on and may be read straight against them.
    """
    one_cycle: bool = False
    ema: bool = False
    intensity_aug: bool = False
    weight_floor: bool = False
    normalise_by_count: bool = False
    per_label_loss: bool = False
    calibrated_targets: bool = False
    finding_calibration: bool = False
    severity_weight: bool = False
    hold_annotations: bool = False
    hold_annotation_studies: bool = False
    grade_ladder: bool = False
    label_errors: Path | None = None
    unread_targets: Path | None = None
    rank_weight: float = 0.0
    distill_weight: float = 0.0
    feature_weight: float = 0.0
    dissent_weight: float = 0.0
    contrast_weight: float = 0.0
    auc_margin: float = 0.0
    teacher_targets: Path | None = None
    teacher_features: Path | None = None
    side_weight: float = 0.0
    focus_label: str | None = None
    unfocused_weight: float = 0.0
    sample_spread: Path | None = None
    sample_alpha: float = 0.5
    sample_ramp: int = 0
    corpus_oversample: float = 0.0
    """How many times heavier each non-target corpus position is in the draw; 0 is a uniform draw."""
    target_corpus: str = "canonical"
    """The corpus tag treated as the target domain in mixed-corpus sampling."""
    pair_memory: int = 0
    whole_stack: bool = False
    recompute_activations: bool = False
    bf16: bool = True
    lora_rank: int = 0
    compile_encoder: bool = True
    vram_gb: float = 0.0
    report_weight: float = 0.0
    report_vectors: Path | None = None
    patience: int = 0
    """Epochs without holdout improvement before stopping; 0 runs the full schedule."""

    certainty_ramp: int = 0
    """How many epochs it takes for per-cell weights to rise from the confidence floor to their
    full trust. 0 disables the ramp and keeps the table every existing arm trained on."""

    studies: int = 0
    """Corpus positions this run touches at all; 0 is every one of them. A SMOKE knob, never an arm."""

    cache_studies: int = 0
    """How many studies the cache being READ holds; 0 is the full one. Names a different file (knee-35yc)."""

    init: Path | None = None
    init_scope: str = "all"
    """A member on disk whose weights this run STARTS from, in place of a fresh head over a published
    trunk. The second half of LP-FT; see `kneemri.training.warm_start.WarmStart` for the mechanism."""

    unfreeze_last: int = Adaptation().unfreeze_last
    unfreeze_offset: int = Adaptation().unfreeze_offset
    head_lr: float = Adaptation().head_lr
    backbone_lr: float = Adaptation().backbone_lr
    layer_decay: float = Adaptation().layer_decay

    weight_decay: float = Adaptation().weight_decay
    """AdamW's decay on every group. Inherited beside an 8e-6 trunk; the rate has since moved 5x."""

    warmup: float = ONE_CYCLE_WARMUP
    """Share of a one-cycle schedule spent climbing. Inert without --one-cycle."""

    grad_clip: float = UNCLIPPED
    """Ceiling on one step's global gradient norm; 0 is the unbounded step the ledger was measured on."""

    sharpness: float = FLAT
    """Radius of the ball SAM descends from the worst point of; 0 is the ordinary step the ledger ran."""

    pool: str = CLS_MEAN
    slice_pooling: str = CHANNELS
    slice_regions: int = StudyDesign().slice_regions
    through_plane: bool = StudyDesign().through_plane
    slot_pooling: str = ATTENDED
    anatomical_prior: bool = False
    encoder_layer: int = -1
    variant: str = SMALL
    encoder_weights: Path | None = None
    encoder_adapt_depth: int = ALL_BLOCKS
    labels: Path = Path("labels.csv")
    eval_labels: Path | None = None
    out: Path | None = None
    paths: Path = Path("paths.yaml")

    @property
    def recipe(self) -> dict[str, str]:
        """Every flag this run was given, as strings — the one account of what the process parsed.

        HERE rather than on `MemberRecord`, which is where it lived, because the checkpoint is written
        at RUN END and the question this answers is asked at run START. A 12-epoch arm spent 84 minutes
        on the card with nothing in its log naming a single flag it had parsed: the stamp existed, and
        the only way to read it was to wait for the run that was in doubt to finish. Now `train` logs
        this before the corpus is mapped, and `MemberRecord.recipe` stamps the same object, so the log
        line and the checkpoint block cannot disagree — a second spelling is exactly how they would.

        Strings because the checkpoint is loaded with `weights_only=True`, which refuses to unpickle a
        `Path`, and because the other consumer is a JSONL line that must survive `json.dumps`.
        """
        return {name: str(value) for name, value in vars(self).items()}

    @classmethod
    def of_recipe(cls, recipe: Mapping[str, str]) -> "TrainingOptions":
        """The run those strings describe — the INVERSE of `recipe`, and beside it for that reason.

        A finished run's account of itself is a dict of strings, and every reader that wanted a VALUE
        out of one has had to re-derive the type: `budget_residuals` needs the slices, batch, encoder
        and trainable depth of 74 recorded runs to price them against what they measured, and hand-
        mapping four of ninety fields is how a reader comes to disagree with the trainer about what a
        recipe said. Going through the dataclass means the two spellings cannot drift — a field added
        to a run's flags is a field this reads, with no second list to remember.

        A key the dataclass does not have is DROPPED rather than raising: the ledger spans renamed
        flags (`band` became `band_low`/`band_high`), and a reader that refuses an old recipe outright
        can price none of the runs that predate the rename. A field the recipe does not mention keeps
        its default, which is what the run that wrote it got. That last rule is why the defaults are
        built first and CHANGED, rather than passed as a constructor's keywords: a field whose value is
        only known at runtime cannot be typed at a call site, and `replace` is the one door the dataclass
        offers for exactly that.
        """
        hints = get_type_hints(cls)
        stated = {name: cls._value(hints[name], text) for name, text in recipe.items() if name in hints}
        return replace(cls(), **stated)

    @staticmethod
    def _value(hint: object, text: str) -> object:
        """One recipe field as the type its dataclass declares, with `X | None` read as its stated half."""
        if text == NOT_STATED:
            return None
        stated = next((argument for argument in get_args(hint) if argument is not type(None)), hint)
        return _READERS[cast("type", stated)](text)

    @property
    def routing(self) -> Routing:
        """How this run turns series into slots — the scheme it reads and the tie rule it reads under."""
        return Routing(scheme=self.scheme, tie=self.tie)

    @property
    def crop(self) -> Crop:
        """The physical crop this run reads, as one value.

        A property rather than a field because `parse` builds this dataclass straight from argparse's
        dest names, so the two flags have to land as scalars — but everything downstream wants the pair,
        and `member_record` in particular must stamp what the RUN used rather than a module constant.
        """
        return Crop(mm=self.crop_mm, centre=self.centre)

    @property
    def band(self) -> Band:
        """The fraction of each series' depth this run's slices are drawn from.

        Two scalar flags and a property for the reason `crop` gives: argparse's dest names have to land
        as fields, and everything downstream wants the pair — `member_record` above all, which stamped
        the module constant here while stamping the run's own value for every neighbouring rule.
        """
        return Band(low=self.band_low, high=self.band_high)

    @property
    def reading(self) -> Reading:
        """The geometry this run decodes under — its crop and its band, which together name its cache."""
        return Reading(crop=self.crop, band=self.band, spacing=self.spacing)

    def cache(self, directory: Path) -> StudyCache:
        """The arrays this run reads — six axes, resolved in ONE place instead of at every entrypoint.

        `main` and `arm_check` each spelled this constructor out, agreeing with each other and with
        `CacheFlags.cache` by inspection. A seventh axis added to the flags and forgotten in one of them
        is not a crash: it is an arm that trains on pixels its recorded config does not describe, and a
        holdout number filed under a reading nobody ran. The axis list is `StudyCache.of`'s, shared with
        the serve side, so the same addition cannot land here and not there.
        """
        return StudyCache.of(self, directory)

    @property
    def supervision(self) -> Supervision:
        """What this run's two label switches mean, as the value `Targets.pair` takes.

        A property for the reason `crop` and `band` give, and for one more: this mapping was two
        expressions inside `main`, which is the only part of the flag-to-policy wiring no test can
        reach — `options_to_policies` walks `RunSetup.trainer` and `RunSetup.model`, and a switch that
        maps to a DATA argument instead of a policy object sat outside that walk. Both of these change
        what the model is fitted on while leaving every recorded config agreeing with itself, which is
        the exact shape of the nine defects this project has already paid for.
        """
        return Supervision(
            floor=CONFIDENCE_FLOOR if self.weight_floor else 0.0,
            calibrated=self.calibrated_targets,
            finding=self.finding_calibration,
            severity=self.severity_weight,
            annotations=not self.hold_annotations,
            withheld=self.hold_annotation_studies,
            suspect=self.label_errors,
            unread=self.unread_targets,
        )

    @classmethod
    def of(cls, argv: list[str]) -> "TrainingOptions":
        """Read an arm's flags into an options object without going through the process's own argv.

        THE COST MODEL HAS TO BE ASKABLE ABOUT A RECIPE THE ASKER IS NOT RUNNING. `vram_need` derives
        an arm's reservation by parsing `sys.argv`, which is the only shape a `$(...)` caller in bash
        needs — and it is exactly the shape `arm_smoke` could not use, because it holds thirty-four
        arms' flag strings in one process and none of them is its own command line. So the smoke sweep
        started arms without ever asking what they would reserve, and one 28.26 GB smoke went onto a
        card already holding 16.24 GB of a live run: it could not allocate, sat the full ten-minute
        bound, and cost the run beside it 734 s for an epoch that had taken 248 s and 255 s.
        """
        return cls(**vars(FlagVocabulary.parser().parse_args(argv)))

    @classmethod
    def parse(cls) -> "TrainingOptions":
        """Read the command line into an options object.

        The VOCABULARY is `FlagVocabulary`'s and this is the only reach for it, which is the split: a
        field here has a default in two homes, the dataclass and the flag, and every test that asserts
        they agree goes through the parser directly rather than through this.
        """
        return cls.of(sys.argv[1:])

    @property
    def positions(self) -> FitPositions:
        """How much of the fold's split this run fits on — the smoke cap and the scaling fraction.

        A property for the reason `crop` and `band` give: the two flags have to land as scalars because
        `parse` builds this dataclass straight from argparse's dest names, and everything downstream
        wants them as one value. The RULES live beside `ReportFolds`, whose split they cut.
        """
        return FitPositions(studies=self.studies, train_fraction=self.train_fraction, split_seed=self.split_seed)
