"""The epoch loop: passes over the corpus, and the decision of which pass is KEPT.

`TrainingOptions.fitted_model` and `fitted_trainer` were pulled out of `train.main` because a run's
SETUP and a run's LOOP are two jobs and only the first grows with every new flag. The loop itself was
left inside `main`, where nothing could call it — and it is the half that decides what a run
PRODUCES rather than how it is configured:

- the score that ranks an epoch is measured under the weights that epoch would SAVE, not the live
  ones, so a run with an EMA cannot rank epochs for a model it never writes to disk;
- a run with no holdout (`--fold -1`) has no score to rank by, so the epoch INDEX stands in and the
  existing keep-the-best machinery keeps the LAST state — with the reported score coming back NaN,
  because a final fit that advertised `holdout 6.0` would beat every real member in `strongest()`;
- the best weights reach the disk the moment they improve, under the same assembler the final save
  uses, so a killed run leaves a member that is submittable and not merely inspectable;
- patience ends the loop without compressing the schedule, which was built for the full epoch count.

Every one of those is a silent-wrong-answer failure rather than a crash, and none of them was
reachable from a test: `main` needs a parsed command line, a mapped cache, a downloaded encoder and a
configured data root before it reaches the first epoch.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from kneemri.holdout_record import HoldoutRecord
from kneemri.member_record import MemberRecord
from kneemri.patience import Patience
from kneemri.run_metrics import EpochRow, RunMetrics
from kneemri.scores import LabelScores
from kneemri.study_cache import CachedCorpus
from kneemri.targets import Targets
from kneemri.trainer import Trainer

logger = logging.getLogger(__name__)

NO_HOLDOUT = float("nan")
"""What a run with nothing held back may report: not a number, because it cannot measure one."""


@dataclass(frozen=True)
class Fold:
    """What a run fits on and what it is scored by — one object because the two must not drift.

    `targets` and `evaluation` are separate tables on purpose: holding the ruler fixed while the
    training labels vary is the only way two label arms are comparable, and `Targets.pair` produces
    the two together for exactly that reason.
    """

    corpus: CachedCorpus
    targets: Targets
    evaluation: Targets
    training: NDArray[np.int64]
    holdout: NDArray[np.int64]

    @property
    def scored(self) -> bool:
        """Whether anything was held back to rank epochs by."""
        return len(self.holdout) > 0


@dataclass(frozen=True)
class RunPlan:
    """How many passes, how large a step, in what order, and where the result is written."""

    destination: Path
    epochs: int = 8
    batch: int = 12
    patience: int = 0
    """Epochs without improvement before stopping; 0 runs the full schedule."""

    seed: int = 2026
    """Data ORDER only — the model's own initialisation is seeded by the caller before it is built."""


@dataclass(frozen=True)
class Fitted:
    """The epoch a run KEPT: its weights, the score it may report, and how it read its holdout."""

    state: dict[str, torch.Tensor]
    holdout: float
    epoch: int
    witness: dict[str, object] | None


class Fitting:
    """Runs the epochs and keeps the best one, writing it out every time it improves."""

    def __init__(self, trainer: Trainer, record: MemberRecord, plan: RunPlan) -> None:
        self.trainer = trainer
        self.record = record
        self.plan = plan
        self.metrics = RunMetrics(plan.destination)
        self.predictions = HoldoutRecord(plan.destination)
        # Written HERE rather than at the first scored epoch, so a run killed before it scores still
        # leaves a file that says what it was trying.
        self.metrics.describe(record.recipe)
        self.scores = LabelScores()
        # Built here rather than passed in: `main` seeded one generator and handed it to nothing else,
        # so owning it costs no behaviour and removes an argument that could be seeded twice.
        self.generator = np.random.default_rng(plan.seed)
        self.patience = Patience(plan.patience)

    def ranked(self, fold: Fold, epoch: int) -> tuple[float, dict[str, float], NDArray[np.float64] | None]:
        """The number this epoch is ranked by, its per-label scores, and what it predicted.

        MUST be called inside `Trainer.evaluated`, together with the state clone it ranks — scoring the
        live weights while saving the averaged ones ranks epochs for a model that is never written.

        With no holdout the epoch INDEX stands in, which rises every time and so makes keep-the-best
        keep the last state. That is the right choice for a final fit and needs no second code path.
        """
        if not fold.scored:
            return float(epoch), {}, None
        predicted = self.trainer.predict(fold.corpus, fold.holdout, self.plan.batch, self.generator)
        per_label = self.scores.per_label(predicted, fold.evaluation.scores[fold.holdout])
        return float(np.nanmean(list(per_label.values()))), per_label, predicted

    def keep(self, fold: Fold, epoch: int, ranked: float, predicted: NDArray[np.float64] | None) -> Fitted:
        """Take this epoch's weights, and put them on disk before the next one can lose them.

        Holding the only copy in memory until the final epoch is how a run that has already produced
        its best epoch loses everything. The witness travels with the weights it was produced by, so a
        checkpoint kept from a killed run carries its own proof of how it should be read.
        """
        model = self.trainer.model
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        # NaN, not `ranked`: with no holdout that value is the epoch index standing in for a score.
        holdout = ranked if fold.scored else NO_HOLDOUT
        witness = None if predicted is None else self.record.witness(fold.targets.identifiers, fold.holdout, predicted)
        if predicted is not None:
            # The SAME array the ledger's per-label numbers were read off, kept whole. Eight studies of
            # it become the witness; the other 873 are what any paired interval over this arm needs, and
            # re-deriving them later would land within a tolerance wider than the interval itself.
            self.predictions.write(
                HoldoutRecord.named(fold.targets.identifiers, fold.holdout),
                predicted,
                fold.evaluation.scores[fold.holdout],
                epoch,
            )
        kept = Fitted(state=state, holdout=holdout, epoch=epoch, witness=witness)
        self.trainer.save_partial(self.plan.destination, self.record.payload(state, holdout, epoch, witness))
        return kept

    def run(self, fold: Fold) -> Fitted:
        """Fit every epoch, and return the one that is kept."""
        best, kept = -1.0, None
        for epoch in range(self.plan.epochs):
            started = time.time()
            loss = self.trainer.fit_epoch(fold.corpus, fold.targets, fold.training, self.plan.batch, self.generator)
            # Both the score and the state come from inside this block, so the number that ranks an
            # epoch is measured on exactly the weights that epoch would write.
            with self.trainer.evaluated():
                ranked, per_label, predicted = self.ranked(fold, epoch)
                improved = ranked > best
                if improved:
                    best, kept = ranked, self.keep(fold, epoch, ranked, predicted)
            seconds = round(time.time() - started)
            logger.info("epoch %d: loss %.4f  holdout mean AUC %.4f  (%ds)", epoch, loss, ranked, seconds)
            # AFTER the metrics row, so a stopped run's ledger is complete up to the epoch it stopped on.
            self.metrics.record(EpochRow(epoch=epoch, loss=loss, auc=ranked, seconds=seconds, per_label=per_label))
            if self.patience.exhausted(improved=improved):
                logger.info(
                    "stopping at epoch %d: %d epochs without improving on %.4f", epoch, self.patience.epochs, best
                )
                break
        if kept is None:
            message = f"no epoch was kept: the run asked for {self.plan.epochs} epochs"
            raise ValueError(message)
        return kept
