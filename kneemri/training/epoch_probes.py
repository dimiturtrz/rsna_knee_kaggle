"""The epoch row as the merge of independent probes, each the one producer of the fields it names.

`Fitting.epochs` built the row by hand, field by field, beside the loop that produced the numbers, so a
new measurement meant another local, another keyword and another place for two of them to disagree.
Each probe here reads one `EpochReading` and answers with its own fields; the row is their union, and a
new measurement is a new probe. Peak VRAM is NOT a probe: the allocator's peak is reset when it is read,
so `RunMetrics.record` reads it at the one moment that is correct.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from jaxtyping import Float, Int
from numpy.typing import NDArray

from core.findings import TARGETS
from kneemri.evaluation.scores import MIDPOINT
from kneemri.runs.run_record import EpochRow
from kneemri.training.step_trace import UNREAD, StepTrace

ECE_BINS = 15
PROBABILITY_FLOOR = 1e-7
"""Predictions are clipped this far inside (0, 1) before a log, so one saturated cell cannot read as infinite loss."""


@dataclass(frozen=True)
class EpochReading:
    """Everything one epoch left behind that its row is read from."""

    epoch: int
    loss: float
    started: float
    fit_seconds: float
    seconds: float
    ranked: float
    per_label: dict[str, float]
    predicted: Float[NDArray[np.float64], "h o"] | None
    truth: Float[NDArray[np.float32], "h o"]
    drawn: Int[NDArray[np.int64], " n"]
    drawn_scores: Float[NDArray[np.float32], "n o"]
    trace: StepTrace

    def row(self, probes: "tuple[EpochProbe, ...]") -> EpochRow:
        """The row this epoch records: each probe's fields merged, no two probes owning the same field."""
        merged: dict[str, object] = {}
        for probe in probes:
            merged.update(probe.read(self))
        return cast("EpochRow", merged)


class EpochProbe(Protocol):
    """One producer of epoch-row fields."""

    def read(self, reading: EpochReading) -> Mapping[str, object]:
        """The fields this probe owns, read off one epoch."""
        ...


class ClockProbe:
    """When the epoch started, how long it took, and how much of that was fitting rather than scoring."""

    def read(self, reading: EpochReading) -> Mapping[str, object]:
        """Epoch index, UNIX start, total and fitting seconds."""
        return {
            "epoch": reading.epoch,
            "started": round(reading.started),
            "seconds": round(reading.seconds),
            "fit_seconds": round(reading.fit_seconds),
        }


class LossProbe:
    """The training loss and the per-study distribution it averages away."""

    def read(self, reading: EpochReading) -> Mapping[str, object]:
        """Mean loss plus `loss_q25` … `loss_q95`."""
        return {"loss": reading.loss} | reading.trace.loss_fields()


class HoldoutProbe:
    """What the holdout says: the ranking AUC, per label, its calibration and its loss."""

    def read(self, reading: EpochReading) -> Mapping[str, object]:
        """`auc`, `per_label`, `ece` and `holdout_loss`; the last two NaN when nothing was held out."""
        held = {"auc": reading.ranked, "per_label": reading.per_label}
        if reading.predicted is None:
            return held | {"ece": UNREAD, "holdout_loss": UNREAD}
        return held | {
            "ece": self.calibration_error(reading.predicted, reading.truth),
            "holdout_loss": self.loss(reading.predicted, reading.truth),
        }

    @staticmethod
    def calibration_error(
        predicted: Float[NDArray[np.float64], "h o"], truth: Float[NDArray[np.float32], "h o"]
    ) -> float:
        """Expected calibration error over every (study, label) cell of the holdout pooled.

        The truth is BINARISED AT 0.5 — `MIDPOINT`, the cut `LabelScores.per_label` scores the twelve AUCs
        against — so the calibration reading is taken against the same partition as the ranking reading.
        Pooled rather than per label, because a per-label ECE is twelve noisy instruments where one will
        do; NaN cells are excluded, since a finding whose truth was never read is not a negative.
        `predicted` is already probabilities: `Trainer.predict` applies the sigmoid.
        """
        cells = ~(np.isnan(predicted) | np.isnan(truth))
        if not cells.any():
            return UNREAD
        probability = predicted[cells]
        positive = (truth[cells] > MIDPOINT).astype(np.float64)
        which = np.minimum((probability * ECE_BINS).astype(np.int64), ECE_BINS - 1)
        count = np.bincount(which, minlength=ECE_BINS).astype(np.float64)
        live = count > 0
        confidence = np.bincount(which, weights=probability, minlength=ECE_BINS)[live] / count[live]
        accuracy = np.bincount(which, weights=positive, minlength=ECE_BINS)[live] / count[live]
        return float((np.abs(confidence - accuracy) * count[live]).sum() / count.sum())

    @staticmethod
    def loss(predicted: Float[NDArray[np.float64], "h o"], truth: Float[NDArray[np.float32], "h o"]) -> float:
        """Mean binary cross-entropy against the GRADED truth, the target the objective itself descends on."""
        cells = ~(np.isnan(predicted) | np.isnan(truth))
        if not cells.any():
            return UNREAD
        probability = np.clip(predicted[cells], PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
        target = truth[cells].astype(np.float64)
        return float(-(target * np.log(probability) + (1 - target) * np.log1p(-probability)).mean())


class OptimProbe:
    """What the optimiser did: pre-clip gradient norms, non-finite steps, the rates it ended on."""

    def read(self, reading: EpochReading) -> Mapping[str, object]:
        """`grad_norm_mean`, `grad_norm_max`, `nonfinite_steps`, `lr`."""
        return reading.trace.optimised()


class FitProbe:
    """How well the model fit the studies it trained on, as it trained on them."""

    def read(self, reading: EpochReading) -> Mapping[str, object]:
        """`fit_auc`."""
        return {"fit_auc": reading.trace.fit_auc()}


class DrawProbe:
    """What the sampler drew: rows, distinct studies, and positives per finding at the midpoint."""

    def read(self, reading: EpochReading) -> Mapping[str, object]:
        """`drawn_rows`, `drawn_studies`, `drawn_positives`."""
        positives = (reading.drawn_scores > MIDPOINT).sum(axis=0)
        return {
            "drawn_rows": len(reading.drawn),
            "drawn_studies": len(np.unique(reading.drawn)),
            "drawn_positives": {name: int(count) for name, count in zip(TARGETS, positives, strict=False)},
        }


PROBES: tuple[EpochProbe, ...] = (ClockProbe(), LossProbe(), HoldoutProbe(), OptimProbe(), FitProbe(), DrawProbe())
