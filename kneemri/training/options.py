"""What one training run's command line SAYS, as a value: `kneemri.training.options`.

Separated from `train.py` (which runs a fit) and from `run_setup.py` (which builds what a fit needs)
because the three are different jobs and only this one is imported by things that never train —
`arm_check` parses a queued invocation with it, `switch_coverage` diffs its fields against the arms,
`member_record` stamps it into a checkpoint. Those consumers were importing the training entrypoint,
and with it `transformers`, to read a dataclass.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.crop_centre import CROP_MM, DEFAULT_CENTRE, Crop
from core.data.corpus.folds import FitPositions
from core.data.corpus.slot_reader import IMAGE_SIZE, SLICE_BAND, Band, Reading
from core.data.corpus.slots import CANONICAL_SCHEME, FIRST_MAXIMAL, Routing
from core.data.corpus.study_cache import StudyCache
from core.data.labels.targets import CONFIDENCE_FLOOR, Supervision
from core.findings import TARGETS
from kneemri.models.backbone import ENCODERS, SMALL
from kneemri.models.model import StudyDesign
from kneemri.models.slice_pooling import CHANNELS, SLICE_POOLINGS
from kneemri.models.slot_head import ATTENDED, SLOT_POOLINGS
from kneemri.models.token_pooling import CLS_MEAN, POOL_PARTS
from kneemri.training.pretrained_encoder import ALL_BLOCKS
from kneemri.training.trainer import Adaptation


@dataclass(frozen=True)
class TrainingOptions:
    """Everything one run varies, parsed once.

    The parser lived in `main`, where it grew a statement per flag and pushed the function over the
    complexity gate twice in one session. A flag is configuration, not control flow, so it belongs in a
    type — and a type can then be constructed in a test without a command line.
    """

    fold: int = 0
    folds: int = 5
    epochs: int = 8
    batch: int = 12
    accumulate: int = 1
    """Micro-batches per optimiser step: the EFFECTIVE batch is `batch * accumulate`, at the activation
    memory of `batch` alone. 1 is the loop every arm in the ledger ran."""
    slices: int = 12
    image_size: int = IMAGE_SIZE
    train_size: int = 0
    """0 means read the cache at its own resolution; anything else resamples to that size."""
    scheme: str = CANONICAL_SCHEME
    crop_mm: float = CROP_MM
    centre: str = DEFAULT_CENTRE
    band_low: float = SLICE_BAND[0]
    band_high: float = SLICE_BAND[1]
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
    severity_weight: bool = False
    hold_annotations: bool = False
    grade_ladder: bool = False
    rank_weight: float = 0.0
    distill_weight: float = 0.0
    dissent_weight: float = 0.0
    auc_margin: float = 0.0
    teacher_targets: Path | None = None
    side_weight: float = 0.0
    focus_label: str | None = None
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

    studies: int = 0
    """Corpus positions this run touches at all; 0 is every one of them. A SMOKE knob, never an arm."""

    cache_studies: int = 0
    """How many studies the cache being READ holds; 0 is the full one. Names a different file (knee-35yc)."""

    unfreeze_last: int = Adaptation().unfreeze_last
    head_lr: float = Adaptation().head_lr
    backbone_lr: float = Adaptation().backbone_lr
    layer_decay: float = Adaptation().layer_decay
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
        return Reading(crop=self.crop, band=self.band)

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
            severity=self.severity_weight,
            annotations=not self.hold_annotations,
        )

    @classmethod
    def parse(cls) -> "TrainingOptions":
        """Read the command line into an options object."""
        return cls(**vars(cls.parser().parse_args()))

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        """The flags, separable from reading them, so a test can ask what argv accepts.

        Every field here has a default in TWO homes — the dataclass and the flag — and a drift between
        them is silent: the tests construct this object directly, so a parser default that moves means
        the command line fits a recipe no test has ever checked. Handing the parser out is what lets
        that be asserted rather than assumed.
        """
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--fold",
            type=int,
            default=0,
            help="-1 trains on EVERY study: no fold is ever assigned -1, so the holdout comes out "
            "empty and nothing is held back. That is the final-fit mode the public recipes use, and it "
            "cannot report a score by construction - the last epoch is kept rather than the best one.",
        )
        parser.add_argument("--folds", type=int, default=5)
        parser.add_argument("--epochs", type=int, default=8)
        parser.add_argument("--batch", type=int, default=12)
        parser.add_argument(
            "--accumulate",
            type=int,
            default=1,
            help="micro-batches summed into one optimiser step. The effective batch is --batch times "
            "this, at the activation memory of --batch alone, which is what lets a wider encoder hold "
            "the settled recipe's batch: dinov2-base at 336 px and 20 slices asks 56.3 GB at batch 12 "
            "and 28.3 at 6. Costs one forward/backward per micro-batch, and restores the GRADIENT only "
            "- anything pooled within a step still sees --batch studies.",
        )
        # EVERY FLAG THAT NAMES A CACHE, from the one place that knows the set. These were eight
        # blocks here and eight more in `build_cache`, and the two had already diverged on the band.
        CacheFlags.add_to(parser)
        parser.add_argument(
            "--train-size",
            type=int,
            default=0,
            help="resolution the model actually READS, resampled from the cache. 0 means the cache's "
            "own. Decouples resolution from decoding: a 448 px cache can be read at 392 without a "
            "second 45-minute build, and without changing the batch or the slice count to afford it.",
        )
        parser.add_argument(
            "--studies",
            type=int,
            default=0,
            help="cap the corpus positions this run touches (0 = all of them). A SMOKE knob: it proves "
            "a recipe wires end to end — model built, loss stepped, checkpoint written — in minutes "
            "instead of holding the card for forty to find out a flag was miswired. The holdout keeps "
            "the fold's ratio so both halves of the loop run, and the score it reports means nothing.",
        )
        parser.add_argument("--seed", type=int, default=2026, help="model init and data order ONLY")
        parser.add_argument(
            "--split-seed",
            type=int,
            default=2026,
            help="fold assignment. SEPARATE from --seed on purpose: deriving the split from the "
            "training seed gives every seed a different holdout, so two runs are not comparable and "
            "cross-scoring them leaks training data.",
        )
        parser.add_argument(
            "--calibrated-targets",
            action="store_true",
            help="replace each report level with what it was MEASURED to be worth against the 57 "
            "annotated studies: 0.08->0.041, 0.28->0.136, 0.68->0.378, 0.82->0.688, 0.94->0.958. Four of "
            "the five assigned values sit outside their own 95%% interval, all too high. Applies to the "
            "TRAINING table only - recalibrating the ruler would raise the AUC by construction.",
        )
        parser.add_argument("--one-cycle", action="store_true", help="attach a OneCycleLR schedule")
        parser.add_argument(
            "--patience",
            type=int,
            default=0,
            help="stop after this many epochs without improving the holdout (0 = run every epoch). "
            "The twelve-epoch cycle peaks at epoch 6 and never recovers, and the best epoch is the one "
            "whose weights are kept, so the tail produces checkpoints that are discarded. The SCHEDULE "
            "is still built for the full count - this ends the loop, it does not compress the anneal.",
        )
        parser.add_argument(
            "--vram-gb",
            type=float,
            default=0.0,
            help="hard cap on this run's share of the card, in GB (0 = 80%% of it, the band the arms "
            "converged on; naming a number also makes the run WAIT for that much to be free). Two "
            "reasons for capping at all. It lets "
            "another project hold the rest of the GPU. And it converts the WORST failure mode we have "
            "into the best one: Windows pages GPU memory over PCIe rather than raising OOM, so a run "
            "that does not fit crawls at a quarter of the power budget while reporting 100%% "
            "utilisation. Under a cap, PyTorch raises OOM immediately instead.",
        )
        # `--compile-encoder` is GONE the way `--bf16` went: compilation is the default, and the flag
        # that used to switch it on now accepts a value nothing reads. Twenty-two of thirty-six arms
        # were running eager for no reason but not naming `$THROUGHPUT` — 21% of every step, on the
        # bulk of the arithmetic, thrown away by an omission rather than by a decision.
        parser.add_argument(
            "--eager-encoder",
            dest="compile_encoder",
            action="store_false",
            help="skip torch.compile on the encoder. THE DEFAULT IS COMPILED: measured -21%% per step "
            "(150 ms eager against 119 ms) for one 2.2 s compilation per process, and score-neutral — "
            "the throughput pair scored 0.7604 plain against 0.7606 compiled, a fifth of the noise "
            "floor. Compiled in place (`nn.Module.compile`), so parameter names are unchanged and the "
            "checkpoint loads through the inference path. This escape exists for the throughput "
            "CONTROLS, which have to run eager to measure what compiling is worth, and for a machine "
            "with no Triton — upstream ships it everywhere except Windows, where `triton-windows` is "
            "pinned in pyproject for exactly this.",
        )
        parser.add_argument(
            "--lora-rank",
            type=int,
            default=0,
            help="adapt every block along a rank-r subspace with the pretrained weights FROZEN, "
            "instead of rewriting the last few blocks. 0 disables it. The base weight stays "
            "bit-identical, so what a small noisy dataset can do to a 1.7B-image representation is "
            "bounded by the rank rather than by the learning rate.",
        )
        # `--bf16` is GONE, not deprecated. It existed only so fourteen arms written before bf16 became
        # the default would still parse; those arms now take the recipe from experiments/recipe.sh and
        # none of them names it. A flag that accepts the default and does nothing is a thing to keep in
        # step forever, and this repository has one dev and no live system — the old invocation is in
        # git if it is ever wanted.
        parser.add_argument(
            "--fp16",
            dest="bf16",
            action="store_false",
            help="train in fp16 with a loss scaler instead of the bf16 default. THE DEFAULT IS bf16: it "
            "matches fp16's matmul throughput on this card (native from sm_80) and removes scale "
            "collapse, since bf16 carries fp32's exponent range and needs no scaler. Serving is a "
            "SEPARATE choice and is fp16 — a T4 has fp16 tensor cores and none for bf16, 65 TFLOPS "
            "against 8.1, and fp16 has MORE mantissa than bf16, so the only thing given up is range. "
            "Measured on real studies, our members peak at 3%% of fp16's ceiling, and `Ensemble.arithmetic` "
            "does the swap while `predict_member` redoes any non-finite pass in fp32.",
        )
        parser.add_argument(
            "--recompute-activations",
            action="store_true",
            help="gradient checkpointing: 19%% less throughput (measured here) for nearly all of the encoder's "
            "activation memory. Needed once a step puts many images through the encoder - whole-stack "
            "reading and higher resolutions both overflow 32 GB, and an overflow THRASHES rather than "
            "failing, which looks like 100%% utilisation at a quarter of the power budget.",
        )
        parser.add_argument(
            "--ema",
            action="store_true",
            help="score and save an exponential moving average of the weights rather than the live "
            "ones. Universal in the public notebooks, and aimed at the 0.0062 seed spread we measured: "
            "it stops the answer depending on which step training happened to stop at.",
        )
        parser.add_argument("--intensity-aug", action="store_true", help="gamma/blur/contrast/noise too")
        parser.add_argument("--weight-floor", action="store_true", help="an unmentioned finding pulls at 0.25")
        parser.add_argument("--normalise-by-count", action="store_true", help="loss .mean() not /weight-sum")
        parser.add_argument(
            "--severity-weight",
            action="store_true",
            help="a cell pulls at how often its GRADE was confirmed against the image annotations "
            "(0.378 mild / 0.688 moderate / 0.958 severe) rather than at the reader's __conf, which is "
            "0.95 on all three. Values are untouched — this is the weight, not the target.",
        )
        parser.add_argument(
            "--hold-annotations",
            action="store_true",
            help="withhold the 58 image-read annotations from the TRAINING table, so this member's "
            "predictions on those studies never saw their radiologist labels. That is the one input "
            "`kneemri.evaluation.supervision_gap` needs and no member we fit can give it: the override "
            "weights them at 3.0, so every other member is scored against a ruler it trained on. The "
            "holdout ruler is re-read WITH them, so this arm stays comparable to every ledger row.",
        )
        parser.add_argument(
            "--grade-ladder",
            action="store_true",
            help="read the reader's five levels as four ORDERED THRESHOLDS crossed by one latent score "
            "rather than as five values a logit must land on. Order is all an AUC scores, so the plain "
            "loss spends the axis it reads on absolute probabilities the metric never sees. Thresholds "
            "are derived from the measured agreement rates, and nothing downstream changes: the same "
            "per-label logit is stored and served.",
        )
        parser.add_argument(
            "--per-label-loss",
            action="store_true",
            help="normalise the loss PER LABEL so each of the twelve contributes equally, as the metric "
            "does. Pooled, a label's share of the gradient tracks how often it is mentioned — Effusion "
            "is positive in 60%% of studies and Fracture in 7%% — while the score averages the twelve "
            "the same.",
        )
        parser.add_argument(
            "--rank-weight",
            type=float,
            default=0.0,
            help="add a pairwise ranking term at this strength (0 = cross-entropy only). The metric is "
            "a mean of per-label AUCs, which reads ORDER and discards calibration; this term pays only "
            "for pairs of studies ordered the wrong way round.",
        )
        parser.add_argument(
            "--distill-weight",
            type=float,
            default=0.0,
            help="add a pairwise term at this strength asking the student to reproduce an ENSEMBLE's "
            "ordering of these studies (0 = off). Not the usual KL on softened probabilities, which "
            "targets calibration the metric discards. The reader emits five levels, so a pair of studies "
            "on the SAME level teaches the label term nothing — and a finding present in 7%% of studies "
            "strands roughly 87%% of its pairs that way. A continuous teacher orders all of them, so the "
            "gain should concentrate on the rarest findings. Requires --teacher-targets.",
        )
        parser.add_argument(
            "--dissent-weight",
            type=float,
            default=0.0,
            help="add a pairwise ranking term at this strength whose pairs are re-weighted toward the "
            "ones an ENSEMBLE orders WRONG (0 = off). The same teacher file --distill-weight reads, for "
            "the opposite purpose: not to reproduce the pool's ordering but to spend the member's "
            "capacity where that ordering fails, which is the one way to raise ensemble ambiguity "
            "without raising member error. Anchored to the labels throughout — a pair is only ever paid "
            "for in the direction the reports order it. Requires --teacher-targets.",
        )
        parser.add_argument(
            "--auc-margin",
            type=float,
            default=0.0,
            help="REPLACE cross-entropy with the AUC-margin surrogate at this margin width (0 = off, "
            "1.0 = the published default and full separation on a sigmoid scale). Cross-entropy "
            "optimises calibration and the metric is twelve rank statistics, so this drops the part of "
            "the answer the scoreboard cannot read rather than adding a term to compete with it. Won "
            "CheXpert 2020 (Yuan et al., arXiv 2012.03173). Composed with none of the terms above, "
            "which all add themselves to the cross-entropy this removes.",
        )
        parser.add_argument(
            "--pair-memory",
            type=int,
            default=0,
            help="keep this many past steps' readouts so the ranking term can form pairs ACROSS batches "
            "(0 = within-batch only). The metric orders the whole test set, while a batch of 12 can only "
            "ask for the ordering of the twelve studies that happened to be loaded together; batch size "
            "here is set by VRAM, so without this the pair pool is chosen by the card. Requires "
            "--rank-weight, since it changes that term and nothing else.",
        )
        parser.add_argument(
            "--side-weight",
            type=float,
            default=0.0,
            help="add a within-study term at this strength asking the two columns of a compartment pair "
            "(medial/lateral meniscus, medial/lateral OA) to be ordered as the reports grade them. "
            "Cross-entropy scores the twelve columns independently, so a head can minimise it on a pair "
            "by emitting the pair's marginal twice; side_blindness measures the Lateral OA column doing "
            "exactly that, at chance on every member.",
        )
        parser.add_argument(
            "--focus-label",
            default=None,
            choices=TARGETS,
            help="fit ONE of the twelve findings: every other column's loss weight is zeroed, so the "
            "whole encoder goes behind it. The published per-finding numbers this project is measured "
            "against come from SPECIALIST models, while the one published twelve-finding panel averages "
            "0.812 — below our own mean. This is the instrument that says whether our per-finding gaps "
            "are the price of jointness. The head still emits twelve logits, so the checkpoint stays a "
            "member; only the loss narrows.",
        )
        parser.add_argument(
            "--slot-pooling",
            choices=tuple(SLOT_POOLINGS),
            default=ATTENDED,
            help="how the twelve diagnoses read the six slots. 'attend' gives each diagnosis its own "
            "query over the present slots; 'mean' pools them uniformly and is the ABLATION - it exists "
            "to give the attention a denominator, since every number in the ledger was measured on a "
            "recipe that already contained it.",
        )
        parser.add_argument(
            "--pool",
            choices=tuple(POOL_PARTS),
            default=CLS_MEAN,
            help="how a slice's patch tokens become one vector. 'cls_mean' is the class token beside "
            "the patch mean and is what every member on disk was fitted with; 'cls_mean_focal' adds "
            "the mean of each channel's top eighth over the patch grid, which a plain mean dilutes - "
            "the small findings (meniscus, ACL, MCL) are the ones that occupy a fraction of the field.",
        )
        parser.add_argument(
            "--slice-pooling",
            choices=tuple(SLICE_POOLINGS),
            default=CHANNELS,
            help="how a group of slices becomes one feature. 'channels' stacks three slices as RGB, "
            "which couples a group's physical thickness to the slice count; 'max' encodes each slice "
            "on its own and takes the elementwise max, which does not, and is what the .903 reference "
            "does. 'attend' encodes each slice alone as 'max' does and then pools them by learned "
            "attention instead, which keeps WHICH slice mattered - the operation the post-MRNet "
            "literature uses. 'max' and 'attend' cost three encoder passes per group.",
        )
        parser.add_argument(
            "--slice-regions",
            type=int,
            default=StudyDesign().slice_regions,
            help="how many depth regions 'max' pooling takes its maximum within before averaging over "
            "them. 1 is one maximum over the whole band, which is what every member on disk was fitted "
            "with; higher values give each part of the depth its own maximum, so a quiet peripheral "
            "finding stops competing with a loud central one (band_wide and band_dense each bought one "
            "end of the band at the other's expense). Free - no parameters, no extra encoder passes.",
        )
        parser.add_argument("--labels", type=Path, required=True, help="label table to TRAIN on")
        parser.add_argument(
            "--eval-labels",
            type=Path,
            default=None,
            help="table to SCORE against (default: --labels). Hold this fixed across arms when the "
            "training labels are the variable, or the arms are measured with different rulers.",
        )
        cls._encoder_arguments(parser)
        parser.add_argument(
            "--train-fraction",
            type=float,
            default=1.0,
            help="share of the TRAINING studies to fit on, holdout untouched — the data-scaling knob. "
            "Separate from --studies, which cuts both halves so a smoke still exercises the whole loop; "
            "this cuts one side only, so the score is comparable to every full-data row in the ledger.",
        )
        parser.add_argument("--report-vectors", type=Path, default=None, help="npz from report_embeddings")
        parser.add_argument(
            "--teacher-targets",
            type=Path,
            default=None,
            help="npz of an ensemble's scores for the TRAINING studies, read by --distill-weight and by "
            "--dissent-weight. "
            "Its members must not have seen the holdout this arm is scored on: a --fold -1 member saw "
            "every study, so teaching from one leaks the evaluation labels into the student.",
        )
        parser.add_argument("--out", type=Path, default=None)
        parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
        return parser

    @staticmethod
    def _encoder_arguments(parser: argparse.ArgumentParser) -> None:
        """What the encoder IS, what it reads, and what of it adapts — the trunk half of the recipe.

        Split out of `parser` rather than left inline because that method crossed the statement ceiling,
        and this is the group that comes out whole: every flag here is a statement about the trunk, and
        none of them is read by anything that does not already hold an `Adaptation`.
        """
        parser.add_argument(
            "--encoder-layer",
            type=int,
            default=-1,
            help="which encoder block the head reads. -1 is the last, which is what every member so far "
            "was fitted on. The final block of a self-supervised ViT is the most specialised toward its "
            "PRETRAINING objective, and our findings are 1-3 mm - both DINOv2 and DINOv3 ship "
            "get_intermediate_layers for dense work. Costs no extra forward pass.",
        )
        parser.add_argument(
            "--through-plane",
            action="store_true",
            help="give each slice its two depth neighbours as the encoder's other two input channels "
            "instead of three copies of itself. Same slice axis, same pooling, same token / parameter / "
            "FLOP count as the control — the only change is what the three channels hold, so it is the "
            "one arm that puts through-plane information INTO a slice's feature vector rather than "
            "mixing depth after the encoder has already reduced each slice to one.",
        )
        parser.add_argument(
            "--anatomical-prior",
            action="store_true",
            help="start each diagnosis's slot attention from how the finding is actually read",
        )
        parser.add_argument(
            "--variant",
            choices=tuple(ENCODERS),
            default=SMALL,
            help="which pretrained encoder. DINOv2-small is the committed default: DINOv3 is refuted "
            "as an improvement at -0.038 and retained only as a diversity member (see "
            "interpretations/kneemri/2026-08-13_dinov2-stays-and-why-dinov3-loses-the-focal-findings.md).",
        )
        parser.add_argument(
            "--unfreeze-last",
            type=int,
            default=Adaptation().unfreeze_last,
            help="how many of the encoder's trailing blocks train. The restraint is a HYPOTHESIS, not "
            "a measurement: with 58 image-read labels the early generic blocks may have too little "
            "supervision to improve and enough to damage. Capacity is the only axis that has ever "
            "cleared our noise floor, so the hypothesis is worth testing rather than inheriting.",
        )
        parser.add_argument("--head-lr", type=float, default=Adaptation().head_lr)
        parser.add_argument(
            "--backbone-lr",
            type=float,
            default=Adaptation().backbone_lr,
            help="how fast the ENCODER adapts. The 125:1 ratio to the head was inherited, never "
            "measured, and at 8e-6 the encoder barely moves - which is why our depth bracket came out "
            "neutral: unfreezing more blocks cannot help at a rate where no block moves.",
        )
        parser.add_argument(
            "--layer-decay",
            type=float,
            default=Adaptation().layer_decay,
            help="scale the encoder rate DOWN by this factor per block below the top one (0.65-0.75 is "
            "the usual range; 1.0, the default, is the flat rate every ledger number was measured at). "
            "Without it depth and rate can only move together: --unfreeze-last 12 at the rate that makes "
            "the top blocks move applies that same rate to the encoder's most general layers, so a "
            "deeper arm is not a deeper version of the shallow one, it is a different perturbation.",
        )
        parser.add_argument(
            "--whole-stack",
            action="store_true",
            help="read EVERY cached slice each step instead of one sampled 3-slice window. Needs a "
            "per-slice pooling or a tubelet trunk, and for a tubelet trunk it is the only path that "
            "reaches a window wider than 3 at all. A step currently sees 18 of a study's ~180 slices, "
            "while averaging "
            "ten windows at INFERENCE is worth +0.0156 - the largest single effect measured here - so "
            "training was discarding information it already had. Costs about 4x a step.",
        )
        parser.add_argument(
            "--report-weight",
            type=float,
            default=0.0,
            help="also predict each study's REPORT embedding, at this weight (0 = off). Our supervision "
            "is twelve numbers where the report has a median of 129 words in twelve languages; this "
            "term supervises the representation with the rest of it. Training-only - test.csv has no "
            "Report column, so inference is unchanged.",
        )
        parser.add_argument(
            "--encoder-weights",
            type=Path,
            default=None,
            help="start the encoder from a domain-adapted checkpoint written by "
            "kneemri.training.pretrain instead of the stock natural-image weights. Everything else in "
            "the recipe is unchanged, so a run with this flag and its matched control differ in one "
            "variable: what the encoder knew before supervision began.",
        )
        parser.add_argument(
            "--encoder-adapt-depth",
            type=int,
            default=ALL_BLOCKS,
            help="take only the patch embedding and the first N blocks from --encoder-weights, leaving "
            "the rest stock. 0 takes the whole adaptation. The argument for on-domain pretraining only "
            "ever concerned the blocks --unfreeze-last leaves frozen, which supervision cannot write into "
            "and "
            "so cannot repair; the top blocks the head reads are the ones a reconstruction objective "
            "rewrites hardest, and two full loads measured on 2026-08-20 lost 0.17 with block 11 moved "
            "55% and 21% of its own norm.",
        )

    @property
    def positions(self) -> FitPositions:
        """How much of the fold's split this run fits on — the smoke cap and the scaling fraction.

        A property for the reason `crop` and `band` give: the two flags have to land as scalars because
        `parse` builds this dataclass straight from argparse's dest names, and everything downstream
        wants them as one value. The RULES live beside `ReportFolds`, whose split they cut.
        """
        return FitPositions(studies=self.studies, train_fraction=self.train_fraction, split_seed=self.split_seed)
