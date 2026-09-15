"""The training recipe as a tree — the root every knob hangs off, and the parser generated from it.

One declaration per knob (`ConfigLeaf`): its type, its default and its help line sit on the node that
groups it, and the command line is WALKED out of the tree rather than written beside it. Nodes live
beside the module that consumes them where that costs no import cycle (`RegularisationConfig`,
`SamplingConfig`, `TeacherConfig`, `EncoderConfig`); the rest live here. The cache axes are the one
exception to generation: `CacheFlags` already serves `build_cache` and seven other entrypoints, so
`DataConfig` holds those leaves bare and the parser asks `CacheFlags` to spell them.
"""

import argparse
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import Field, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import cast, get_args, get_type_hints

from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.crop_centre import CROP_MM, DEFAULT_CENTRE, Crop
from core.data.corpus.folds import FitPositions
from core.data.corpus.slot_reader import IMAGE_SIZE, SLICE_BAND, SPREAD, Band, Reading
from core.data.corpus.slots import CANONICAL_SCHEME, FIRST_MAXIMAL, Routing
from core.data.corpus.study_cache import StudyCache
from core.data.labels.targets import CONFIDENCE_FLOOR, Supervision
from kneemri.models.model import StudyDesign
from kneemri.models.slice_pooling import CHANNELS, SLICE_POOLINGS
from kneemri.models.slot_head import ATTENDED, SLOT_POOLINGS
from kneemri.models.token_pooling import CLS_MEAN, POOL_PARTS
from kneemri.runs.member_record import RunStamp
from kneemri.training.adaptation import EncoderConfig
from kneemri.training.augmentation import RegularisationConfig
from kneemri.training.config_leaf import FLAG, ConfigLeaf, Spelling
from kneemri.training.gradient_clip import UNCLIPPED
from kneemri.training.recipe_layers import SET, RecipeLayers
from kneemri.training.sampling import SamplingConfig
from kneemri.training.sharpness import FLAT
from kneemri.training.teacher_targets import TeacherConfig
from kneemri.training.trainer import ONE_CYCLE_WARMUP

NOT_STATED = "None"
"""How a recipe spells a field that was never given — `str(None)`, since it stringifies every value.

It is ambiguous for the two `str | None` fields, where the literal text and the absent value have the
same spelling, and unambiguous for every other field: no int, float, Path or bool stringifies to this."""

_READERS: dict[type, Callable[[str], object]] = {
    int: int,
    float: float,
    str: str,
    Path: Path,
    bool: lambda text: text == str(True),
}
"""How each leaf type reads back off a string — `bool` being the one that cannot use its own
constructor, since `bool("False")` is True and would silently arm every switch a run left off."""


@dataclass(frozen=True)
class RunConfig:
    """Which run this is — its fold, its seeds, its files and how it holds the card."""

    fold: int = ConfigLeaf.of(
        default=0,
        says=(
            "-1 trains on EVERY study: no fold is ever assigned -1, so the holdout comes out empty and "
            "nothing is held back. That is the final-fit mode the public recipes use, and it cannot report a "
            "score by construction - the last epoch is kept rather than the best one."
        ),
    )
    folds: int = ConfigLeaf.of(default=5)
    exclude_fold: int = ConfigLeaf.of(
        default=-1,
        says=(
            "a fold dropped from TRAINING as well as from the holdout; -1 excludes nothing. A teacher fitted "
            "for distillation must not have seen the fold its STUDENT is scored on: --fold k alone leaves "
            "fold 0 in every teacher's training set, so the student mimics four functions that each memorised"
            " its own holdout and the fold-0 read is inflated through function mimicry. --exclude-fold 0 is "
            "what makes an out-of-fold teacher out-of-fold."
        ),
    )
    seed: int = ConfigLeaf.of(default=2026, says="model init and data order ONLY")
    split_seed: int = ConfigLeaf.of(
        default=2026,
        says=(
            "fold assignment. SEPARATE from --seed on purpose: deriving the split from the training seed "
            "gives every seed a different holdout, so two runs are not comparable and cross-scoring them "
            "leaks training data."
        ),
    )
    studies: int = ConfigLeaf.of(
        default=0,
        says=(
            "cap the corpus positions this run touches (0 = all of them). A SMOKE knob: it proves a recipe "
            "wires end to end — model built, loss stepped, checkpoint written — in minutes instead of holding"
            " the card for forty to find out a flag was miswired. The holdout keeps the fold's ratio so both "
            "halves of the loop run, and the score it reports means nothing."
        ),
    )
    train_fraction: float = ConfigLeaf.of(
        default=1.0,
        says=(
            "share of the TRAINING studies to fit on, holdout untouched — the data-scaling knob. Separate "
            "from --studies, which cuts both halves so a smoke still exercises the whole loop; this cuts one "
            "side only, so the score is comparable to every full-data row in the ledger."
        ),
    )
    vram_gb: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "hard cap on this run's share of the card, in GB (0 = 80%% of it, the band the arms converged on;"
            " naming a number also makes the run WAIT for that much to be free). Two reasons for capping at "
            "all. It lets another project hold the rest of the GPU. And it converts the WORST failure mode we"
            " have into the best one: Windows pages GPU memory over PCIe rather than raising OOM, so a run "
            "that does not fit crawls at a quarter of the power budget while reporting 100%% utilisation. "
            "Under a cap, PyTorch raises OOM immediately instead."
        ),
    )
    compile_encoder: bool = ConfigLeaf.of(
        default=True,
        says=(
            "skip torch.compile on the encoder. THE DEFAULT IS COMPILED: measured -21%% per step (150 ms "
            "eager against 119 ms) for one 2.2 s compilation per process, and score-neutral — the throughput "
            "pair scored 0.7604 plain against 0.7606 compiled, a fifth of the noise floor. Compiled in place "
            "(`nn.Module.compile`), so parameter names are unchanged and the checkpoint loads through the "
            "inference path. This escape exists for the throughput CONTROLS, which have to run eager to "
            "measure what compiling is worth, and for a machine with no Triton — upstream ships it everywhere"
            " except Windows, where `triton-windows` is pinned in pyproject for exactly this."
        ),
        flag="--eager-encoder",
        action="store_false",
    )
    bf16: bool = ConfigLeaf.of(
        default=True,
        says=(
            "train in fp16 with a loss scaler instead of the bf16 default. THE DEFAULT IS bf16: it matches "
            "fp16's matmul throughput on this card (native from sm_80) and removes scale collapse, since bf16"
            " carries fp32's exponent range and needs no scaler. Serving is a SEPARATE choice and is fp16 — a"
            " T4 has fp16 tensor cores and none for bf16, 65 TFLOPS against 8.1, and fp16 has MORE mantissa "
            "than bf16, so the only thing given up is range. Measured on real studies, our members peak at "
            "3%% of fp16's ceiling, and `Ensemble.arithmetic` does the swap while `predict_member` redoes any"
            " non-finite pass in fp32."
        ),
        flag="--fp16",
        action="store_false",
    )
    labels: Path = ConfigLeaf.of(default=Path("labels.csv"), says="label table to TRAIN on", required=True)
    eval_labels: Path | None = ConfigLeaf.of(
        default=None,
        says=(
            "table to SCORE against (default: --labels). Hold this fixed across arms when the training labels"
            " are the variable, or the arms are measured with different rulers."
        ),
    )
    report_vectors: Path | None = ConfigLeaf.of(default=None, says="npz from report_embeddings")
    out: Path | None = ConfigLeaf.of(default=None)
    paths: Path = ConfigLeaf.of(default=Path("paths.yaml"))
    config: Path | None = ConfigLeaf.of(default=None, says="a recipe TOML; flags on the line override it")

    @property
    def positions(self) -> FitPositions:
        """How much of the fold's split this run fits on — the smoke cap and the scaling fraction.

        The MODEL seed is not in it, as with the fold: only `split_seed` moves which studies exist. The
        rules live beside `ReportFolds`, whose split they cut.
        """
        return FitPositions(studies=self.studies, train_fraction=self.train_fraction, split_seed=self.split_seed)


@dataclass(frozen=True)
class DataConfig:
    """The arrays a run reads; every axis but `train_size` is spelled by `CacheFlags`."""

    slices: int = 12
    image_size: int = IMAGE_SIZE
    scheme: str = CANONICAL_SCHEME
    crop_mm: float = CROP_MM
    centre: str = DEFAULT_CENTRE
    band_low: float = SLICE_BAND[0]
    band_high: float = SLICE_BAND[1]
    spacing: str = SPREAD
    tie: str = FIRST_MAXIMAL
    cache_studies: int = 0
    train_size: int = ConfigLeaf.of(
        default=0,
        says=(
            "resolution the model actually READS, resampled from the cache. 0 means the cache's own. "
            "Decouples resolution from decoding: a 448 px cache can be read at 392 without a second 45-minute"
            " build, and without changing the batch or the slice count to afford it."
        ),
    )

    @property
    def routing(self) -> Routing:
        """How this run turns series into slots — the scheme it reads and the tie rule it reads under."""
        return Routing(scheme=self.scheme, tie=self.tie)

    @property
    def crop(self) -> Crop:
        """The physical crop this run reads — what `member_record` stamps rather than a module constant."""
        return Crop(mm=self.crop_mm, centre=self.centre)

    @property
    def band(self) -> Band:
        """The fraction of each series' depth this run's slices are drawn from, as the stamped pair."""
        return Band(low=self.band_low, high=self.band_high)

    @property
    def reading(self) -> Reading:
        """The geometry this run decodes under — its crop and its band, which together name its cache."""
        return Reading(crop=self.crop, band=self.band, spacing=self.spacing)

    def cache(self, directory: Path) -> StudyCache:
        """The arrays this run reads, resolved through `StudyCache.of` — the axis list the serve side shares.

        `main` and `arm_check` each spelled this constructor out once. A seventh axis forgotten in one of
        them is not a crash: it is an arm training on pixels its recorded config does not describe.
        """
        return StudyCache.of(self, directory)


@dataclass(frozen=True)
class OptimConfig:
    """The loop — how long, how big a step, and when an epoch is kept."""

    epochs: int = ConfigLeaf.of(default=8)
    batch: int = ConfigLeaf.of(default=12)
    accumulate: int = ConfigLeaf.of(
        default=1,
        says=(
            "micro-batches summed into one optimiser step. The effective batch is --batch times this, at the "
            "activation memory of --batch alone, which is what lets a wider encoder hold the settled recipe's"
            " batch: dinov2-base at 336 px and 20 slices asks 56.3 GB at batch 12 and 28.3 at 6. Costs one "
            "forward/backward per micro-batch, and restores the GRADIENT only - anything pooled within a step"
            " still sees --batch studies."
        ),
    )
    one_cycle: bool = ConfigLeaf.of(default=False, says="attach a OneCycleLR schedule")
    ema: bool = ConfigLeaf.of(
        default=False,
        says=(
            "score and save an exponential moving average of the weights rather than the live ones. Universal"
            " in the public notebooks, and aimed at the 0.0062 seed spread we measured: it stops the answer "
            "depending on which step training happened to stop at."
        ),
    )
    patience: int = ConfigLeaf.of(
        default=0,
        says=(
            "stop after this many epochs without improving the holdout (0 = run every epoch). The twelve-"
            "epoch cycle peaks at epoch 6 and never recovers, and the best epoch is the one whose weights are"
            " kept, so the tail produces checkpoints that are discarded. The SCHEDULE is still built for the "
            "full count - this ends the loop, it does not compress the anneal."
        ),
    )
    patience_from_progress: float = ConfigLeaf.of(default=0.0, says="share before keeping starts")
    certainty_ramp: int = ConfigLeaf.of(
        default=0,
        says=(
            "SUPERSEDED by --hard-*, kept for ledger comparability. Epochs over which per-cell weights rise "
            "from the confidence floor to their full trust (0 = off). Early epochs therefore treat every cell"
            " as equally uncertain, and later epochs recover the full certainty-weighted table. The total "
            "weight mass is held constant so the loss scale matches the un-ramped control."
        ),
    )
    grad_clip: float = ConfigLeaf.of(
        default=UNCLIPPED,
        says=(
            "ceiling on the global gradient norm of one optimiser step (0 = unbounded, which is what every "
            "number in the ledger was measured under). The failure it addresses is measured: native224_long "
            "slid -0.0165 over twelve extra epochs while its training loss kept falling, which is the shape a"
            " few large late steps produce. Clipping bounds how far one batch may move the weights; it does "
            "not change where the walk is allowed to end up, which is --weight-decay. 1.0 is the usual value "
            "in the ViT recipes this project's trunk came from."
        ),
    )
    sharpness: float = ConfigLeaf.of(
        default=FLAT,
        says=(
            "radius of the ball sharpness-aware minimisation descends from the worst point of (0 = the "
            "ordinary step, which is what every number in the ledger was measured under). It doubles the "
            "forward and backward cost of a step, and it needs --accumulate 1. The gap it addresses is the "
            "one every closed axis here shares: the corner reads ~0.865 held out while the board reads 0.944,"
            " and aggregation, readout, slice depth, model scale, corpus, labels and weight-space averaging "
            "all closed by moving the offline number without moving the transfer. This changes the flatness "
            "of the minimum rather than what is read. 0.05 is the usual value in the ViT recipes this "
            "project's trunk came from."
        ),
    )
    warmup: float = ConfigLeaf.of(
        default=ONE_CYCLE_WARMUP,
        says=(
            "share of the one-cycle schedule spent warming UP before the anneal begins (0.15 is the inherited"
            " default and what every ledger number was measured at). Over twelve epochs that is 1.8 epochs "
            "climbing to the peak rate on a run whose best epoch is 7, so a seventh of the budget is spent "
            "below the rate the recipe was tuned for. Read with --one-cycle only — without a schedule there "
            "is no warmup to size, and the flag is then inert."
        ),
    )


@dataclass(frozen=True)
class ModelConfig:
    """How the head reads the trunk's tokens, slices and slots."""

    pool: str = ConfigLeaf.of(
        default=CLS_MEAN,
        says=(
            "how a slice's patch tokens become one vector. 'cls_mean' is the class token beside the patch "
            "mean and is what every member on disk was fitted with; 'cls_mean_focal' adds the mean of each "
            "channel's top eighth over the patch grid, which a plain mean dilutes - the small findings "
            "(meniscus, ACL, MCL) are the ones that occupy a fraction of the field."
        ),
        choices=tuple(POOL_PARTS),
    )
    slice_pooling: str = ConfigLeaf.of(
        default=CHANNELS,
        says=(
            "how a group of slices becomes one feature. 'channels' stacks three slices as RGB, which couples "
            "a group's physical thickness to the slice count; 'max' encodes each slice on its own and takes "
            "the elementwise max, which does not, and is what the .903 reference does. 'attend' encodes each "
            "slice alone as 'max' does and then pools them by learned attention instead, which keeps WHICH "
            "slice mattered - the operation the post-MRNet literature uses. 'max' and 'attend' cost three "
            "encoder passes per group."
        ),
        choices=tuple(SLICE_POOLINGS),
    )
    slice_regions: int = ConfigLeaf.of(
        default=StudyDesign().slice_regions,
        says=(
            "how many depth regions 'max' pooling takes its maximum within before averaging over them. 1 is "
            "one maximum over the whole band, which is what every member on disk was fitted with; higher "
            "values give each part of the depth its own maximum, so a quiet peripheral finding stops "
            "competing with a loud central one (band_wide and band_dense each bought one end of the band at "
            "the other's expense). Free - no parameters, no extra encoder passes."
        ),
    )
    slot_pooling: str = ConfigLeaf.of(
        default=ATTENDED,
        says=(
            "how the twelve diagnoses read the six slots. 'attend' gives each diagnosis its own query over "
            "the present slots; 'mean' pools them uniformly and is the ABLATION - it exists to give the "
            "attention a denominator, since every number in the ledger was measured on a recipe that already "
            "contained it. 'attend_fixed' is the attention again with the anatomical prior held as a BUFFER "
            "rather than as a warm start, so a gradient step cannot erase it - the form aprior_dense's "
            "washout says was never actually tested."
        ),
        choices=tuple(SLOT_POOLINGS),
    )


@dataclass(frozen=True)
class TargetConfig:
    """What the model is fitted TO — which labels, weighted how, under which extra losses."""

    calibrated_targets: bool = ConfigLeaf.of(
        default=False,
        says=(
            "replace each report level with what it was MEASURED to be worth against the 57 annotated "
            "studies: 0.08->0.041, 0.28->0.136, 0.68->0.378, 0.82->0.688, 0.94->0.958. Four of the five "
            "assigned values sit outside their own 95%% interval, all too high. Applies to the TRAINING table"
            " only - recalibrating the ruler would raise the AUC by construction."
        ),
    )
    finding_calibration: bool = ConfigLeaf.of(
        default=False,
        says=(
            "on top of --calibrated-targets, shift each finding's level by its own measured offset. The "
            "pooled map assumes a 0.68 is worth 0.378 on every finding; over the reader's positive calls the "
            "radiologist agrees 44.4%% of the time on Contusion and 81.2%% on Synovitis, and the two fail in "
            "OPPOSITE directions. Offsets are empirical-Bayes shrunk (tau=0.294), so a thin finding keeps the"
            " pooled value. REQUIRES --calibrated-targets."
        ),
    )
    severity_weight: bool = ConfigLeaf.of(
        default=False,
        says=(
            "a cell pulls at how often its GRADE was confirmed against the image annotations (0.378 mild / "
            "0.688 moderate / 0.958 severe) rather than at the reader's __conf, which is 0.95 on all three. "
            "Values are untouched — this is the weight, not the target."
        ),
    )
    hold_annotations: bool = ConfigLeaf.of(
        default=False,
        says=(
            "withhold the 58 image-read annotations from the TRAINING table, so this member's predictions on "
            "those studies never saw their radiologist labels. That is the one input "
            "`kneemri.evaluation.supervision_gap` needs and no member we fit can give it: the override "
            "weights them at 3.0, so every other member is scored against a ruler it trained on. The holdout "
            "ruler is re-read WITH them, so this arm stays comparable to every ledger row. It withholds the "
            "annotator's VALUE and NOT the study: all 58 rows still train at their report-derived values, 45 "
            "of them in the fold-0 training split. For a gold read that is actually out of sample, this flag "
            "is not enough — add --hold-annotation-studies."
        ),
    )
    hold_annotation_studies: bool = ConfigLeaf.of(
        default=False,
        says=(
            "withhold the annotated STUDIES themselves: every cell of all 58 is zeroed in the WEIGHT array, "
            "so they contribute nothing to the loss. Requires --hold-annotations, which it implies rather "
            "than replaces. This is what every header claiming a 'clean' or 'held out' gold reading said it "
            "was doing and was not (knee-x4muo); it costs 1.66% of the training corpus, and at 13 usable "
            "studies the ruler it buys may still be too thin to read."
        ),
    )
    grade_ladder: bool = ConfigLeaf.of(
        default=False,
        says=(
            "read the reader's five levels as four ORDERED THRESHOLDS crossed by one latent score rather than"
            " as five values a logit must land on. Order is all an AUC scores, so the plain loss spends the "
            "axis it reads on absolute probabilities the metric never sees. Thresholds are derived from the "
            "measured agreement rates, and nothing downstream changes: the same per-label logit is stored and"
            " served."
        ),
    )
    label_errors: Path | None = ConfigLeaf.of(
        default=None,
        says=(
            "a two-column table of (StudyInstanceUID, target) cells an independent signal calls mislabelled; "
            "each is given zero loss weight. Written by `kneemri.evaluation.label_errors --flags`, whose "
            "confident joint SIZES its own flag set, so no threshold is fitted here. An annotated cell is "
            "never pruned - the radiologist's reading is the truth the detector was gated against."
        ),
    )
    unread_targets: Path | None = ConfigLeaf.of(
        default=None,
        says=(
            "a three-column table of (StudyInstanceUID, target, value) replacing the SCORE on cells the "
            "report never read - the 27.3%% of pairs carrying the constant 0.28 at verdict UNK. The weight is"
            " untouched, so this changes what those cells pull toward and not how hard. The complement of "
            "--label-errors, which reaches the cell the reader got wrong rather than the cell the reader left"
            " silent. An annotated cell is never overridden."
        ),
    )
    weight_floor: bool = ConfigLeaf.of(default=False, says="an unmentioned finding pulls at 0.25")
    normalise_by_count: bool = ConfigLeaf.of(default=False, says="loss .mean() not /weight-sum")
    per_label_loss: bool = ConfigLeaf.of(
        default=False,
        says=(
            "normalise the loss PER LABEL so each of the twelve contributes equally, as the metric does. "
            "Pooled, a label's share of the gradient tracks how often it is mentioned — Effusion is positive "
            "in 60%% of studies and Fracture in 7%% — while the score averages the twelve the same."
        ),
    )
    rank_weight: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "add a pairwise ranking term at this strength (0 = cross-entropy only). The metric is a mean of "
            "per-label AUCs, which reads ORDER and discards calibration; this term pays only for pairs of "
            "studies ordered the wrong way round."
        ),
    )
    contrast_weight: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "add a per-finding supervised-contrastive term at this strength over what each diagnosis READ "
            "from the study, before the readout collapsed it to a logit (0 = off). Every other term here is a"
            " function of the logits — a dot product against one learned direction — so two studies read for "
            "different reasons can share it. This one asks the representation itself to put torn beside torn "
            "and intact beside intact, per finding, which is the only phrasing this repo has for 'is the "
            "tissue CONNECTED' rather than 'how bright, and where'. Composes with --rank-weight, which the "
            "house recipe always passes, so a contrastive arm stays a single-variable change against its "
            "control; not composed with --distill-weight, --dissent-weight or --pair-memory, whose dispatch "
            "returns a different objective."
        ),
    )
    auc_margin: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "REPLACE cross-entropy with the AUC-margin surrogate at this margin width (0 = off, 1.0 = the "
            "published default and full separation on a sigmoid scale). Cross-entropy optimises calibration "
            "and the metric is twelve rank statistics, so this drops the part of the answer the scoreboard "
            "cannot read rather than adding a term to compete with it. Won CheXpert 2020 (Yuan et al., arXiv "
            "2012.03173). Composed with none of the terms above, which all add themselves to the cross-"
            "entropy this removes."
        ),
    )
    pair_memory: int = ConfigLeaf.of(
        default=0,
        says=(
            "keep this many past steps' readouts so the ranking term can form pairs ACROSS batches (0 = "
            "within-batch only). The metric orders the whole test set, while a batch of 12 can only ask for "
            "the ordering of the twelve studies that happened to be loaded together; batch size here is set "
            "by VRAM, so without this the pair pool is chosen by the card. Requires --rank-weight, since it "
            "changes that term and nothing else."
        ),
    )
    side_weight: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "add a within-study term at this strength asking the two columns of a compartment pair "
            "(medial/lateral meniscus, medial/lateral OA) to be ordered as the reports grade them. Cross-"
            "entropy scores the twelve columns independently, so a head can minimise it on a pair by emitting"
            " the pair's marginal twice; side_blindness measures the Lateral OA column doing exactly that, at"
            " chance on every member."
        ),
    )
    focus_label: str | None = ConfigLeaf.of(
        default=None,
        says=(
            "lean the loss on one or more of the twelve findings, joined by '+' in ONE quoted token (a "
            "finding's name holds a space): --focus-label \"MCL+Lateral OA\". Every unnamed column's loss "
            "weight is multiplied by --unfocused-weight, which at its default of 0 is the hard mask. The head"
            " still emits twelve logits, so the checkpoint stays a member; only the loss leans. An unknown "
            "name RAISES rather than falling through to the plain recipe."
        ),
    )
    unfocused_weight: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "what an unnamed finding's loss weight is multiplied by, 0 masking it outright. The mask has been"
            " run and it failed in a named way: focus_mcl regressed its own target label by -0.0576 paired "
            "and collapsed to a 0.5332 twelve-column mean, because one supervisory signal per study is not "
            "enough to fit an encoder on. A weight strictly inside (0, 1) keeps all twelve signals and only "
            "moves the gradient's balance."
        ),
    )

    @property
    def supervision(self) -> Supervision:
        """What the label switches mean, as the value `Targets.pair` takes.

        These change what the model is FITTED ON while every recorded config keeps agreeing with itself,
        and they map to a data argument rather than a policy object, so `options_to_policies` cannot see
        them — this property is where a test reaches them.
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


@dataclass(frozen=True)
class TrainConfig:
    """Everything one run varies, as a tree — built from argv, a recipe file or a checkpoint's recipe."""

    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    regularisation: RegularisationConfig = field(default_factory=RegularisationConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)

    @classmethod
    def leaves(cls) -> Iterator[tuple[Field[object], object]]:
        """Every knob under the root with its declared type, depth-first — the walk every reader shares."""
        return ((leaf, hint) for _, leaf, hint in TrainConfig._under(cls, ""))

    @classmethod
    def addresses(cls) -> dict[str, str]:
        """Every leaf's dotted path to its name (`optim.epochs` to `epochs`) — how a recipe layer names it."""
        return {path: leaf.name for path, leaf, _ in TrainConfig._under(cls, "")}

    @staticmethod
    def _under(kind: type, prefix: str) -> Iterator[tuple[str, Field[object], object]]:
        hints = get_type_hints(kind)
        for leaf in fields(kind):
            hint = hints[leaf.name]
            if is_dataclass(hint):
                yield from TrainConfig._under(cast("type", hint), f"{prefix}{leaf.name}.")
            else:
                yield f"{prefix}{leaf.name}", leaf, hint

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        """The training command line, generated: one flag per leaf that declares a `Spelling`."""
        parser = argparse.ArgumentParser()
        CacheFlags.add_to(parser)
        parser.add_argument(
            SET,
            action="append",
            default=argparse.SUPPRESS,
            metavar="PATH=VALUE",
            help="one leaf by dotted path, over every --config",
        )
        for leaf, hint in cls.leaves():
            if FLAG in leaf.metadata:
                TrainConfig._declare(parser, leaf, hint)
        return parser

    @staticmethod
    def _declare(parser: argparse.ArgumentParser, leaf: Field[object], hint: object) -> None:
        spelling = cast("Spelling", dict(leaf.metadata[FLAG]))
        flag = spelling.pop("flag", f"--{leaf.name.replace('_', '-')}")
        if hint is bool:
            spelling.setdefault("action", "store_true")
            parser.add_argument(flag, dest=leaf.name, **spelling)
            return
        parser.add_argument(flag, dest=leaf.name, type=TrainConfig._reader(hint), default=leaf.default, **spelling)

    @classmethod
    def parse(cls) -> "TrainConfig":
        """The process's own command line as a tree."""
        return cls.of(sys.argv[1:])

    @classmethod
    def of(cls, argv: list[str]) -> "TrainConfig":
        """The tree an arm's flags describe: defaults, then each `--config`, then `--set`, then the flags.

        From a LIST, so a caller can price a recipe it is not itself running: `arm_smoke` holds thirty-four
        arms' lines in one process, and a 28.26 GB smoke once went onto a card holding a live run because
        nothing could ask what an arm that was not the process would reserve.
        """
        parser = cls.parser()
        parser.set_defaults(**cls.of_layers(RecipeLayers.of(argv)))
        return cls.of_flat(vars(parser.parse_args(argv)))

    @classmethod
    def of_flat(cls, values: Mapping[str, object]) -> "TrainConfig":
        """The tree holding these leaf values by name; a leaf the mapping does not name keeps its default."""
        return cast("TrainConfig", TrainConfig._built(cls, values))

    @staticmethod
    def _built(kind: type, values: Mapping[str, object]) -> object:
        hints = get_type_hints(kind)
        nodes = {leaf.name: cast("type", hints[leaf.name]) for leaf in fields(kind) if is_dataclass(hints[leaf.name])}
        stated = {name: TrainConfig._built(node, values) for name, node in nodes.items()}
        return kind(**stated, **{leaf.name: values[leaf.name] for leaf in fields(kind) if leaf.name in values})

    def with_leaves(self, **values: object) -> "TrainConfig":
        """This tree with the named leaves changed, wherever in the tree they hang."""
        return self.of_flat({**self.flat, **values})

    @property
    def flat(self) -> dict[str, object]:
        """Every leaf by name — the tree read back flat, the shape recipes and checkpoints are keyed by."""
        return dict(TrainConfig._values(self))

    @property
    def recipe(self) -> dict[str, str]:
        """Every leaf as strings — the one account of what the process parsed, logged at start and stamped.

        FLAT, as it was before the tree, so every checkpoint and metrics line on disk reads the way it
        always did. Strings because a checkpoint is loaded with `weights_only=True`, which refuses to
        unpickle a `Path`, and because the other consumer is a JSONL line that must survive `json.dumps`.
        """
        return {name: str(value) for name, value in self.flat.items()}

    @property
    def stamp(self) -> RunStamp:
        """What a checkpoint records about this run beyond its recipe — the values a rebuild must know."""
        data, model, encoder = self.data, self.model, self.encoder
        return RunStamp(
            recipe=self.recipe,
            image_size=data.image_size,
            crop=data.crop,
            band=data.band,
            spacing=data.spacing,
            routing=data.routing,
            train_size=data.train_size,
            cache_studies=data.cache_studies,
            slices=data.slices,
            pool=model.pool,
            slice_pooling=model.slice_pooling,
            slice_regions=model.slice_regions,
            slot_pooling=model.slot_pooling,
            encoder_layer=encoder.encoder_layer,
            through_plane=encoder.through_plane,
            anatomical_prior=encoder.anatomical_prior,
            lora_rank=encoder.lora_rank,
            report_weight=encoder.report_weight,
            whole_stack=encoder.whole_stack,
            unfreeze_last=encoder.unfreeze_last,
            unfreeze_offset=encoder.unfreeze_offset,
            head_lr=encoder.head_lr,
            backbone_lr=encoder.backbone_lr,
            layer_decay=encoder.layer_decay,
            ema=self.optim.ema,
            bf16=self.run.bf16,
            rank_weight=self.targets.rank_weight,
            pair_memory=self.targets.pair_memory,
        )

    @classmethod
    def of_recipe(cls, recipe: Mapping[str, str]) -> "TrainConfig":
        """The run those strings describe — the inverse of `recipe`.

        A key naming no leaf is DROPPED rather than raising: the ledger spans renamed flags (`band` became
        `band_low`/`band_high`), and a reader refusing an old recipe could price none of the runs before
        the rename. A leaf the recipe does not mention keeps its default, which the run that wrote it got.
        """
        hints = {leaf.name: hint for leaf, hint in cls.leaves()}
        return cls.of_flat({name: cls.typed(hints[name], text) for name, text in recipe.items() if name in hints})

    @classmethod
    def spelled(cls, argv: list[str]) -> list[str]:
        """The line with every leaf its recipe layers set spelled as the flag it stands for, ahead of it.

        For checks that read what an arm STATED rather than what it parsed to: a key in a recipe file
        is written down exactly as a flag on the line is, and must not read as silence.
        """
        return [f"--{name.replace('_', '-')}" for name in cls.of_layers(RecipeLayers.of(argv))] + argv

    @staticmethod
    def _values(node: object) -> Iterator[tuple[str, object]]:
        for leaf in fields(cast("type", node)):
            value = vars(node)[leaf.name]
            if is_dataclass(value):
                yield from TrainConfig._values(value)
            else:
                yield leaf.name, value

    @classmethod
    def of_layers(cls, layers: RecipeLayers) -> dict[str, object]:
        """The values a line's recipe layers state, typed as the leaves they name — what its flags override.

        NESTED TOML keyed by node (`[optim]` then `epochs = 36`), so a recipe is written once and an arm
        names it with `--config` instead of re-typing thirty flags; `--set optim.epochs=4` is the same
        address on the line.
        """
        hints = {leaf.name: hint for leaf, hint in cls.leaves()}
        return {name: cls.typed(hints[name], str(value)) for name, value in layers.values(cls.addresses()).items()}

    @staticmethod
    def typed(hint: object, text: str) -> object:
        """One leaf's text as the type it declares, with `X | None` read as its stated half."""
        if text == NOT_STATED:
            return None
        return TrainConfig._reader(hint)(text)

    @staticmethod
    def _reader(hint: object) -> Callable[[str], object]:
        stated = next((argument for argument in get_args(hint) if argument is not type(None)), hint)
        return _READERS[cast("type", stated)]
