"""Fit one member: `python -m kneemri.train --fold 0 --epochs 8`.

Writes a checkpoint in the same shape the published package uses — weights under `model` beside the
config, the fold scores and a fingerprint — so our own members and theirs load through one code path.
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from core.paths import DataRoot
from kneemri.backbone import ENCODERS, SMALL
from kneemri.folds import ReportFolds
from kneemri.member_record import MemberRecord
from kneemri.model import StudyDesign, StudyModel
from kneemri.objective import Objective, RankedObjective
from kneemri.report_supervision import ReportSupervision
from kneemri.run_metrics import EpochRow, RunMetrics
from kneemri.scores import LabelScores
from kneemri.slot_reader import IMAGE_SIZE
from kneemri.slots import SCHEMES
from kneemri.study_cache import StudyCache
from kneemri.targets import CONFIDENCE_FLOOR, Targets
from kneemri.trainer import Adaptation, AugmentConfig, DeviceBudget, Trainer

GROUP = 3

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
    scheme: str = StudyCache.CANONICAL
    seed: int = 2026
    split_seed: int = 2026
    one_cycle: bool = False
    ema: bool = False
    intensity_aug: bool = False
    weight_floor: bool = False
    normalise_by_count: bool = False
    rank_weight: float = 0.0
    whole_stack: bool = False
    recompute_activations: bool = False
    compile_encoder: bool = False
    vram_gb: float = 0.0
    report_weight: float = 0.0
    report_vectors: Path | None = None
    unfreeze_last: int = Adaptation().unfreeze_last
    head_lr: float = Adaptation().head_lr
    backbone_lr: float = Adaptation().backbone_lr
    slice_pooling: str = "channels"
    anatomical_prior: bool = False
    variant: str = SMALL
    labels: Path = Path("labels.csv")
    eval_labels: Path | None = None
    out: Path | None = None
    paths: Path = Path("paths.yaml")

    @classmethod
    def parse(cls) -> "TrainingOptions":
        """Read the command line into an options object."""
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
        parser.add_argument("--slices", type=int, default=12)
        parser.add_argument(
            "--image-size",
            type=int,
            default=IMAGE_SIZE,
            help="resolution of the cache to train on. Selects the cache file AND is recorded in the "
            "member's config and fingerprint, so a member can never advertise a reading it was not "
            "fitted under.",
        )
        parser.add_argument(
            "--train-size",
            type=int,
            default=0,
            help="resolution the model actually READS, resampled from the cache. 0 means the cache's "
            "own. Decouples resolution from decoding: a 448 px cache can be read at 392 without a "
            "second 45-minute build, and without changing the batch or the slice count to afford it.",
        )
        parser.add_argument(
            "--scheme",
            choices=tuple(SCHEMES),
            default=StudyCache.CANONICAL,
            help="slot scheme. 'extended' routes three sequences the canonical six cannot express - "
            "coronal PD without fat suppression, axial T1, axial fluid without fat suppression - which "
            "are 96%% of the series we currently discard. It needs its own cache and produces members "
            "with a wider slot embedding, so it does not interoperate with the published six-slot ones.",
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
        parser.add_argument("--one-cycle", action="store_true", help="attach a OneCycleLR schedule")
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
            help="torch.compile the encoder with dynamic shapes. The encoder only, because training "
            "calls supervised() rather than forward(); dynamic because the last batch of an epoch is "
            "partial and a static graph would recompile every epoch.",
        )
        parser.add_argument(
            "--recompute-activations",
            action="store_true",
            help="gradient checkpointing: ~30%% more compute for nearly all of the encoder's "
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
            "--rank-weight",
            type=float,
            default=0.0,
            help="add a pairwise ranking term at this strength (0 = cross-entropy only). The metric is "
            "a mean of per-label AUCs, which reads ORDER and discards calibration; this term pays only "
            "for pairs of studies ordered the wrong way round.",
        )
        parser.add_argument(
            "--slice-pooling",
            choices=("channels", "max", "attend"),
            default="channels",
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
            "--anatomical-prior",
            action="store_true",
            help="start each diagnosis's slot attention from how the finding is actually read",
        )
        parser.add_argument(
            "--variant",
            choices=tuple(ENCODERS),
            default=SMALL,
            help="DINOv2 size. base is 86M parameters and 768-wide against small's 22M and 384.",
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
        return cls(**vars(parser.parse_args()))

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
            return RankedObjective(self.rank_weight, normalise_by_count=self.normalise_by_count)
        return Objective(normalise_by_count=self.normalise_by_count)

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
        if self.whole_stack:
            trainer.whole_stack(self.slices)
        if self.train_size:
            trainer.resample_to(self.train_size)
        if self.report_weight and self.report_vectors:
            trainer.report_supervision(
                ReportSupervision.from_file(self.report_vectors, identifiers, self.report_weight)
            )
        if self.ema:
            trainer.exponential_average()
        if self.one_cycle:
            # ceil, not floor: our batcher KEEPS the final partial batch, so a floor undercounts by one
            # step per epoch and OneCycleLR raises on the very last step of the run — after every epoch has
            # been logged and before the checkpoint is written, which is the most expensive place to fail.
            steps_per_epoch = -(-training_positions // self.batch)
            trainer.one_cycle(self.epochs * max(steps_per_epoch, 1))
        return trainer


def main() -> None:
    arguments = TrainingOptions.parse()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    torch.manual_seed(arguments.seed)
    generator = np.random.default_rng(arguments.seed)
    device = DeviceBudget(arguments.vram_gb).select()

    root = DataRoot.from_config(arguments.paths)
    cache = StudyCache(
        root.processed(), slices=arguments.slices, image_size=arguments.image_size, scheme=arguments.scheme
    )
    corpus, identifiers = cache.load()
    targets, evaluation = Targets.pair(
        arguments.labels,
        arguments.eval_labels,
        root.raw() / "train.csv",
        identifiers,
        CONFIDENCE_FLOOR if arguments.weight_floor else 0.0,
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

    design = StudyDesign(slice_pooling=arguments.slice_pooling, anatomical_prior=arguments.anatomical_prior)
    encoder = ENCODERS[arguments.variant]
    slots = len(SCHEMES[arguments.scheme])
    model = StudyModel(AutoModel.from_pretrained(encoder.repository), encoder.dim, slots, design)
    if arguments.recompute_activations:
        model.recompute_activations()
    if arguments.compile_encoder:
        model.compile_encoder()
    scores = LabelScores()
    trainer = arguments.fitted_trainer(model, device, identifiers, len(training))

    destination = arguments.out or (root.processed() / f"member_fold{arguments.fold}.pt")
    metrics = RunMetrics(destination)
    record = MemberRecord(arguments, encoder, model, device, [slot.name for slot in SCHEMES[arguments.scheme]])
    best, best_state = -1.0, None
    witnessed: dict[str, object] | None = None
    for epoch in range(arguments.epochs):
        started = time.time()
        mean_loss = trainer.fit_epoch(corpus, targets, training, arguments.batch, generator)
        # An empty holdout (--fold -1) has no score to rank epochs by, so the epoch number stands in:
        # it increases every time, which makes the existing "keep the best" machinery keep the LAST
        # state instead. That is the right choice for a final fit and needs no second code path.
        empty: dict[str, float] = {}
        # Both the score and the state come from inside this block, so the number that ranks an epoch is
        # measured on exactly the weights that epoch would write.
        with trainer.evaluated():
            predicted = trainer.predict(corpus, holdout, arguments.batch, generator) if len(holdout) else None
            per_label = empty if predicted is None else scores.per_label(predicted, evaluation.scores[holdout])
            mean = float(np.nanmean(list(per_label.values()))) if per_label else float(epoch)
            if mean > best:
                best = mean
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                # NaN, not `mean`: with no holdout `mean` is the epoch index standing in for a score.
                scored = mean if per_label else float("nan")
                # The witness travels with the weights it was produced by, so a checkpoint kept from a
                # killed run carries its own proof of how it should be read.
                seen = None if predicted is None else record.witness(identifiers, holdout, predicted)
                trainer.save_partial(destination, record.payload(best_state, scored, epoch, seen))
                witnessed = seen
        logger.info(f"epoch {epoch}: loss {mean_loss:.4f}  holdout mean AUC {mean:.4f}  ({time.time() - started:.0f}s)")
        metrics.record(
            EpochRow(epoch=epoch, loss=mean_loss, auc=mean, seconds=round(time.time() - started), per_label=per_label)
        )

    # A run with no holdout ranks its epochs by epoch NUMBER, so `best` is an epoch index rather than a
    # score. Recording that would have a final-fit member advertise holdout 6.0 — beating every real
    # member in `strongest()`, and reading as a score in the manifest and the ledger. NaN is the honest
    # value: this run cannot report one by construction.
    reported = best if len(holdout) else float("nan")
    model.load_state_dict(best_state)  # pyrefly: ignore[bad-argument-type]
    # The SAME assembler the periodic save uses, so a run killed at its best epoch and a run allowed
    # to finish produce checkpoints of identical shape.
    # best_state is set on the first epoch that scores; a run with none is already unusable above.
    torch.save(record.payload(best_state, reported, arguments.epochs - 1, witnessed), destination)  # pyrefly: ignore[bad-argument-type]
    logger.info(
        "best holdout mean AUC %.4f -> %s | %s",
        reported,
        destination,
        json.dumps({"fold": arguments.fold, "holdout": reported}),
    )


if __name__ == "__main__":
    main()
