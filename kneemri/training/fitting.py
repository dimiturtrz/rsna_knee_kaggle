"""The epoch loop: passes over the corpus, and the decision of which pass is KEPT.

`RunSetup.model` and `RunSetup.trainer` were pulled out of `train.main` because a run's
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

from core.data.corpus.study_cache import CachedCorpus
from core.data.labels.targets import Targets
from core.wakelock import Wakelock
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.runs.holdout_record import HoldoutRecord
from kneemri.runs.member_record import WITNESS_STUDIES, MemberRecord
from kneemri.runs.run_metrics import EpochRow, RunMetrics
from kneemri.training.patience import Patience
from kneemri.training.trainer import Trainer

logger = logging.getLogger(__name__)

NO_HOLDOUT = float("nan")
"""What a run with nothing held back may report: not a number, because it cannot measure one."""

ECE_BINS = 15
"""How many equal-width bins over [0, 1] an epoch's calibration error is read over.

Fifteen is the conventional count — Guo et al.'s and every replication's since — and nothing here has
measured a better one. What matters is that it is FIXED: two epochs' ECEs are the same instrument read
twice, which is the only way the number can rank anything.
"""


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
        # leaves a file that says what it was trying — and it OPENS the record rather than appending to
        # it, so this run's epochs cannot be read as a continuation of whatever ran under the name before.
        self.metrics.begin(record.recipe)
        self.scores = LabelScores()
        # Built here rather than passed in: `main` seeded one generator and handed it to nothing else,
        # so owning it costs no behaviour and removes an argument that could be seeded twice.
        self.generator = np.random.default_rng(plan.seed)
        self.patience = Patience(plan.patience)

    @staticmethod
    def _calibration_error(predicted: NDArray[np.float64], truth: NDArray[np.float32]) -> float:
        """One epoch's expected calibration error, over every (study, label) cell of the holdout pooled.

        The truth is BINARISED AT 0.5 — `MIDPOINT`, the cut `LabelScores.per_label` scores the twelve AUCs
        against — so the calibration reading is taken against the same partition as the ranking reading.
        Pooled rather than per label, because a per-label ECE is twelve noisy instruments where one will
        do: the bins are filled by every cell, with NaN cells excluded — a finding whose truth was never
        read is not a negative.

        `predicted` is already probabilities in [0, 1]: `Trainer.predict` applies the sigmoid, and this
        relies on that rather than repeating it.
        """
        cells = ~(np.isnan(predicted) | np.isnan(truth))
        if not cells.any():
            return float("nan")
        probability = predicted[cells]
        positive = (truth[cells] > MIDPOINT).astype(np.float64)
        which = np.minimum((probability * ECE_BINS).astype(np.int64), ECE_BINS - 1)
        count = np.bincount(which, minlength=ECE_BINS).astype(np.float64)
        live = count > 0
        confidence = np.bincount(which, weights=probability, minlength=ECE_BINS)[live] / count[live]
        accuracy = np.bincount(which, weights=positive, minlength=ECE_BINS)[live] / count[live]
        return float((np.abs(confidence - accuracy) * count[live]).sum() / count.sum())

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
        witness = self.witnessed(fold, predicted)
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

    def witnessed(self, fold: Fold, predicted: NDArray[np.float64] | None) -> dict[str, object]:
        """A replay fingerprint for these weights, which a run with NO HOLDOUT must also carry.

        A witness is an end-to-end claim — re-read these named studies through the inference path and
        you must get these numbers back — and nothing in that claim needs the studies to have been held
        out. But the holdout was the only place it was ever taken from, so `--fold -1` produced
        `predicted is None` and silently wrote no witness at all. Five all-data members shipped inside a
        live board entry that way (knee-52mu), and `witness_check` reported them as "trained before the
        check existed", which is the wrong cause and hid the real one.

        With no holdout the fingerprint is taken from the first training studies instead. They are
        studies the weights have SEEN, so these numbers say nothing about generalisation — and they were
        never asked to. What they pin is the reading: image size, slot list, decode resolution, window,
        precision. Every defect this check has ever caught was a consumer-side default disagreeing with
        a recorded value, and a memorised study exposes that exactly as well as an unseen one.

        The extra forward is free of the run's randomness: `Trainer.batches` takes `shuffle=False` here
        and never draws from the generator, so a member fitted through this path is bit-identical to one
        fitted before it existed.
        """
        if predicted is not None:
            return self.record.witness(fold.targets.identifiers, fold.holdout, predicted, self.plan.batch)
        chosen = fold.training[:WITNESS_STUDIES]
        replayed = self.trainer.predict(fold.corpus, chosen, self.plan.batch, self.generator)
        return self.record.witness(fold.targets.identifiers, chosen, replayed, self.plan.batch)

    def run(self, fold: Fold) -> Fitted:
        """Fit every epoch, and return the one that is kept.

        Held awake for the whole loop, because the idle timer counts INPUT and an unattended run
        provides none: three runs suspended mid-epoch and reported the suspend as several hours of
        epoch (knee-wyzj). Taken here rather than in `main` so it covers exactly the span that must not
        be interrupted, and released by the context manager whether the loop returns or raises.
        """
        with Wakelock():
            return self.epochs(fold)

    def epochs(self, fold: Fold) -> Fitted:
        """Every pass over the corpus, and the one whose weights are kept."""
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
            # `ranked` is an epoch INDEX when there is no holdout, and it is only fit to rank epochs by.
            # Reported as itself it becomes a score: the log reads "mean AUC 4.0000" and the ledger row
            # carries 4.0, which `ledger_check` then quotes as the arm's best — its own docstring says a
            # final fit records NaN, and until now only the checkpoint did.
            scored = ranked if fold.scored else NO_HOLDOUT
            # The SAME partition the AUCs are scored under, so the calibration number and the ranking
            # number describe one binarisation. NaN rather than absent when nothing was held out,
            # matching what a final fit already records in the `auc` column.
            ece = (
                self._calibration_error(predicted, fold.evaluation.scores[fold.holdout])
                if predicted is not None
                else NO_HOLDOUT
            )
            quantiles = self.trainer.loss_quantiles
            # THE DENOMINATOR, WHENEVER IT IS NOT THE COMPETITION'S. `nanmean` drops a finding with only
            # one class present on this holdout, so the headline is a mean over however many survived —
            # and two runs whose means are read against each other may not have averaged the same set.
            # Only said when it differs from twelve, because a line printed every epoch stops being read.
            defined = sum(1 for value in per_label.values() if not np.isnan(value))
            over = "" if defined == len(per_label) else f" over {defined} of {len(per_label)} labels"
            logger.info("epoch %d: loss %.4f  holdout mean AUC %.4f%s  (%ds)", epoch, loss, scored, over, seconds)
            # AFTER the metrics row, so a stopped run's ledger is complete up to the epoch it stopped on.
            self.metrics.record(
                EpochRow(
                    epoch=epoch,
                    loss=loss,
                    auc=scored,
                    ece=ece,
                    loss_q25=quantiles["q25"],
                    loss_q50=quantiles["q50"],
                    loss_q75=quantiles["q75"],
                    loss_q90=quantiles["q90"],
                    loss_q95=quantiles["q95"],
                    seconds=seconds,
                    per_label=per_label,
                    started=round(started),
                )
            )
            if self.patience.exhausted(improved=improved):
                logger.info(
                    "stopping at epoch %d: %d epochs without improving on %.4f", epoch, self.patience.epochs, best
                )
                break
        if kept is None:
            message = f"no epoch was kept: the run asked for {self.plan.epochs} epochs"
            raise ValueError(message)
        return kept
