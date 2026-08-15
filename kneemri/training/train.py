"""Fit one member: `python -m kneemri.training.train --fold 0 --epochs 8`.

Writes a checkpoint in the same shape the published package uses — weights under `model` beside the
config, the fold scores and a fingerprint — so our own members and theirs load through one code path.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel

from core.paths import DataRoot
from kneemri.corpus.cache_flags import CacheFlags
from kneemri.corpus.crop_centre import CROP_MM, DEFAULT_CENTRE, Crop
from kneemri.corpus.folds import ReportFolds
from kneemri.corpus.slot_reader import IMAGE_SIZE, SLICE_BAND, Band, Reading
from kneemri.corpus.slots import CANONICAL_SCHEME, FIRST_MAXIMAL, SCHEMES, Routing
from kneemri.corpus.study_cache import StudyCache
from kneemri.labels.targets import CONFIDENCE_FLOOR, Supervision, Targets
from kneemri.models.backbone import ENCODERS, SMALL, Encoder
from kneemri.models.model import CHANNELS, CLS_MEAN, POOL_PARTS, SLICE_POOLINGS, StudyDesign, StudyModel
from kneemri.models.slot_head import ATTENDED, SLOT_POOLINGS
from kneemri.runs.member_record import MemberRecord
from kneemri.training.fitting import Fitting, Fold, RunPlan
from kneemri.training.objective import Objective, RankedObjective
from kneemri.training.report_supervision import ReportSupervision
from kneemri.training.trainer import Adaptation, AugmentConfig, DeviceBudget, Trainer

logger = logging.getLogger(__name__)


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
    one_cycle: bool = False
    ema: bool = False
    intensity_aug: bool = False
    weight_floor: bool = False
    normalise_by_count: bool = False
    per_label_loss: bool = False
    calibrated_targets: bool = False
    rank_weight: float = 0.0
    whole_stack: bool = False
    recompute_activations: bool = False
    bf16: bool = True
    lora_rank: int = 0
    compile_encoder: bool = False
    vram_gb: float = 0.0
    report_weight: float = 0.0
    report_vectors: Path | None = None
    patience: int = 0
    """Epochs without holdout improvement before stopping; 0 runs the full schedule."""

    unfreeze_last: int = Adaptation().unfreeze_last
    head_lr: float = Adaptation().head_lr
    backbone_lr: float = Adaptation().backbone_lr
    pool: str = CLS_MEAN
    slice_pooling: str = CHANNELS
    slot_pooling: str = ATTENDED
    anatomical_prior: bool = False
    encoder_layer: int = -1
    variant: str = SMALL
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

    @property
    def supervision(self) -> Supervision:
        """What this run's two label switches mean, as the value `Targets.pair` takes.

        A property for the reason `crop` and `band` give, and for one more: this mapping was two
        expressions inside `main`, which is the only part of the flag-to-policy wiring no test can
        reach — `options_to_policies` walks `fitted_trainer` and `fitted_model`, and a switch that
        maps to a DATA argument instead of a policy object sat outside that walk. Both of these change
        what the model is fitted on while leaving every recorded config agreeing with itself, which is
        the exact shape of the nine defects this project has already paid for.
        """
        return Supervision(
            floor=CONFIDENCE_FLOOR if self.weight_floor else 0.0,
            calibrated=self.calibrated_targets,
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
            help="hard cap on this run's share of the card, in GB (0 = uncapped). Two reasons. It lets "
            "another project hold the rest of the GPU. And it converts the WORST failure mode we have "
            "into the best one: Windows pages GPU memory over PCIe rather than raising OOM, so a run "
            "that does not fit crawls at a quarter of the power budget while reporting 100%% "
            "utilisation. Under a cap, PyTorch raises OOM immediately instead.",
        )
        parser.add_argument(
            "--compile-encoder",
            action="store_true",
            help="torch.compile the encoder, shape-specialised. The encoder only, because training "
            "calls supervised() rather than forward(). Measured -20%% per step; the partial last batch "
            "costs one 22 s recompile per process, not per epoch. Needs Triton (triton-windows here).",
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
        parser.add_argument("--labels", type=Path, required=True, help="label table to TRAIN on")
        parser.add_argument(
            "--eval-labels",
            type=Path,
            default=None,
            help="table to SCORE against (default: --labels). Hold this fixed across arms when the "
            "training labels are the variable, or the arms are measured with different rulers.",
        )
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
            "--whole-stack",
            action="store_true",
            help="read EVERY cached slice each step instead of one sampled 3-slice window. Requires "
            "--slice-pooling max. A step currently sees 18 of a study's ~180 slices, while averaging "
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
        parser.add_argument("--report-vectors", type=Path, default=None, help="npz from report_embeddings")
        parser.add_argument("--out", type=Path, default=None)
        parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
        return parser

    def adaptation(self) -> Adaptation:
        """How the encoder adapts: which blocks move and how fast, which are one decision."""
        return Adaptation(head_lr=self.head_lr, backbone_lr=self.backbone_lr, unfreeze_last=self.unfreeze_last)

    def objective(self) -> Objective:
        """The loss policy these flags ask for.

        On the options object rather than in `main` so the choice can be checked without a GPU, a
        corpus or a downloaded encoder. The alternative is finding out that a queued arm was miswired
        after it has already waited forty minutes for the card.
        """
        if self.rank_weight:
            return RankedObjective(
                self.rank_weight, normalise_by_count=self.normalise_by_count, per_label=self.per_label_loss
            )
        return Objective(normalise_by_count=self.normalise_by_count, per_label=self.per_label_loss)

    def reports(self, identifiers: list[str]) -> ReportSupervision | None:
        """The report-embedding supervision this run asked for, or None — and never half of it.

        A weight with no vectors used to be a silent no-op: `if self.report_weight and
        self.report_vectors` skipped the attachment, and the run trained the plain recipe while its
        ledger row, its config and its name all said report supervision. The arm would have been read
        as a refutation of the mechanism rather than of the invocation.
        """
        if not self.report_weight:
            return None
        if self.report_vectors is None:
            message = (
                f"--report-weight {self.report_weight} asks for report supervision and no "
                "--report-vectors was given, so the run would train the plain recipe under that name"
            )
            raise ValueError(message)
        return ReportSupervision.from_file(self.report_vectors, identifiers, self.report_weight)

    def fitted_model(self, encoder: Encoder, slots: int, design: StudyDesign) -> StudyModel:
        """The model this run asked for, with every wrapper applied IN ORDER.

        The sibling of `fitted_trainer`, extracted for the same reason and by the same gate: `main` was
        accumulating a statement per switch, and the run's SETUP is a different job from the run's LOOP.

        THE ORDER IS THE WHOLE VALUE OF THIS METHOD. Adaptation first, because checkpointing and compile
        both WRAP what they are handed — adapters injected after compilation would sit outside the
        compiled graph, and the run would report a speed and a score belonging to two different models.
        """
        model = StudyModel(AutoModel.from_pretrained(encoder.repository), encoder.dim, slots, design)
        if self.lora_rank:
            logger.info("low-rank adaptation: %d layers at rank %d", model.low_rank(self.lora_rank), self.lora_rank)
        if self.recompute_activations:
            model.recompute_activations()
        if self.compile_encoder:
            model.compile_encoder()
        return model

    def fitted_trainer(
        self,
        model: StudyModel,
        device: torch.device,
        identifiers: list[str],
        training_positions: int,
    ) -> Trainer:
        """A `Trainer` with every optional policy this run asked for already attached.

        The policies are attached rather than passed because each is something a run may or may not want,
        and they were accumulating in `main` as a run of `if self.x: trainer.y()` — eight of them by
        the time resampling arrived, which is what pushed `main` past the statement gate. That gate was
        reading a real thing: the run's SETUP and the run's LOOP are two jobs, and only one of them grows
        every time a flag is added.
        """
        trainer = Trainer(
            model,
            device,
            augment_config=AugmentConfig() if self.intensity_aug else AugmentConfig.geometric_only(),
            adaptation=self.adaptation(),
            objective=self.objective(),
        )
        if self.bf16:
            trainer.bfloat16()
        if self.whole_stack:
            trainer.whole_stack(self.slices)
        if self.train_size:
            trainer.resample_to(self.train_size)
        reports = self.reports(identifiers)
        if reports is not None:
            trainer.report_supervision(reports)
        if self.ema:
            trainer.exponential_average()
        if self.one_cycle:
            # FLOOR now, and it was ceil for a real reason that no longer holds: the batcher used to
            # KEEP the final partial batch, so a floor undercounted by one step per epoch and OneCycleLR
            # raised on the very last step of the run — after every epoch had been logged and before the
            # checkpoint was written, the most expensive place to fail. Training now drops that batch so
            # the encoder's graph has ONE shape, and the step count must follow it exactly.
            steps_per_epoch = max(training_positions // self.batch, 1)
            trainer.one_cycle(self.epochs * max(steps_per_epoch, 1))
        return trainer


def main() -> None:
    arguments = TrainingOptions.parse()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # The MODEL's initialisation. The data order is seeded from the same value inside `Fitting`, which
    # is where the only consumer of it lives.
    torch.manual_seed(arguments.seed)

    # THE CACHE BEFORE THE CARD. A reading nothing has decoded is a `FileNotFoundError` on a memory map,
    # and taking the device first meant an arm held the GPU to discover it — the queue slot is spent
    # either way, but the card need not be. Mapping is cheap: no pixels are read here.
    root = DataRoot.from_config(arguments.paths)
    cache = StudyCache(
        root.processed(),
        slices=arguments.slices,
        image_size=arguments.image_size,
        routing=arguments.routing,
        reading=arguments.reading,
    )
    corpus, identifiers = cache.load()
    device = DeviceBudget(arguments.vram_gb).select()
    targets, evaluation = Targets.pair(
        arguments.labels,
        arguments.eval_labels,
        root.raw() / "train.csv",
        identifiers,
        arguments.supervision,
    )
    # The split is a property of the CORPUS, not of this run: it must not move when the model seed
    # does. Deriving it from --seed gave seed 7 a fold-0 holdout sharing only 192 of 909 studies with
    # seed 2026's, so their holdout numbers were never comparable and scoring one on the other's split
    # was scoring a model on 717 studies it had trained on.
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    training, holdout = folds.holdout(folds.assign(str(root.raw() / "train.csv"), identifiers), arguments.fold)
    # ONE call, and %s rather than a second positional: logging treats extra positionals as %-format
    # arguments, so `logger.info("target coverage:", {...})` printed the label and silently dropped the
    # dict. Every run so far has logged the words "target coverage:" and none of the coverage.
    logger.info(
        "%d studies: %d train / %d holdout (fold %d) | target coverage: %s",
        len(identifiers),
        len(training),
        len(holdout),
        arguments.fold,
        {k: round(v, 3) for k, v in targets.coverage().items()},
    )

    design = StudyDesign(
        pool=arguments.pool,
        slice_pooling=arguments.slice_pooling,
        slot_pooling=arguments.slot_pooling,
        anatomical_prior=arguments.anatomical_prior,
        layer=arguments.encoder_layer,
    )
    encoder = ENCODERS[arguments.variant]
    # Before the weights are fetched and the cache is mapped: a read size the patch grid cannot tile is
    # a configuration mistake, and the failure without this is either a crash deep in the encoder or a
    # silent interpolation of the position embeddings onto a grid the weights never saw.
    encoder.require_tiling(arguments.train_size or arguments.image_size)
    model = arguments.fitted_model(encoder, len(SCHEMES[arguments.scheme]), design)
    trainer = arguments.fitted_trainer(model, device, identifiers, len(training))

    destination = arguments.out or (root.processed() / f"member_fold{arguments.fold}.pt")
    record = MemberRecord(arguments, encoder, model, device, [slot.name for slot in SCHEMES[arguments.scheme]])
    plan = RunPlan(
        destination=destination,
        epochs=arguments.epochs,
        batch=arguments.batch,
        patience=arguments.patience,
        seed=arguments.seed,
    )
    kept = Fitting(trainer, record, plan).run(
        Fold(corpus=corpus, targets=targets, evaluation=evaluation, training=training, holdout=holdout)
    )

    model.load_state_dict(kept.state)
    # The SAME assembler the periodic save uses, so a run killed at its best epoch and a run allowed to
    # finish produce checkpoints of identical shape — and the EPOCH is the kept one in both, where the
    # final save used to record the last epoch index and so contradicted its own partial under patience.
    torch.save(record.payload(kept.state, kept.holdout, kept.epoch, kept.witness), destination)
    logger.info(
        "best holdout mean AUC %.4f -> %s | %s",
        kept.holdout,
        destination,
        json.dumps({"fold": arguments.fold, "holdout": kept.holdout}),
    )


if __name__ == "__main__":
    main()
