"""Scoring predictions against targets — the competition's metric, and nothing else.

This lived on `Trainer` and did not belong there: the cohesion gate measured LCOM4=3 on that class, and
the scoring pair shared no state with the training loop. The practical symptom came first — analysis
scripts were calling `Trainer.area_under_curve(None, ...)`, passing `None` as `self`, because the
function needed no trainer and there was nowhere else to put it.

The metric is the unweighted mean of twelve per-label ROC AUCs, so everything here is rank-based. There
is deliberately no calibration, no threshold, and no probability transform anywhere in this module: AUC
is invariant under any strictly increasing map, so such a step could only add a bug.
"""

import numpy as np
from numpy.typing import NDArray

from kneemri.model import TARGETS

MIDPOINT = 0.5
"""Where a graded target is cut so a holdout AUC exists at all — never a submitted threshold."""


class LabelScores:
    """Per-label ROC AUC and its mean, computed by rank rather than by any library."""

    @staticmethod
    def area_under_curve(predictions: NDArray[np.float64], truth: NDArray[np.int64]) -> float:
        """ROC AUC by rank, which is its definition.

        A label with only one class present is undefined rather than zero: reporting 0.5 there would
        silently drag the mean toward chance on exactly the rare findings that matter most.
        """
        positives, negatives = int(truth.sum()), int((1 - truth).sum())
        if positives == 0 or negatives == 0:
            return float("nan")
        order = predictions.argsort().argsort() + 1
        return float((order[truth == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))

    def per_label(self, predictions: NDArray[np.float64], targets: NDArray[np.float32]) -> dict[str, float]:
        """One AUC per finding, against graded targets binarised at their midpoint.

        Binarising exists only so an AUC is defined; it decides which epoch is kept and never a
        submitted score, and the metric is invariant to any increasing map of the predictions anyway.
        """
        return {
            target: self.area_under_curve(
                predictions[:, column].astype(np.float64), (targets[:, column] > MIDPOINT).astype(np.int64)
            )
            for column, target in enumerate(TARGETS)
        }

    def mean(self, predictions: NDArray[np.float64], targets: NDArray[np.float32]) -> float:
        """The competition's score: the unweighted mean over the twelve findings.

        `nanmean`, so a label that is undefined on this split does not poison the whole number — but a
        caller reading only the mean should know how many labels were undefined, which `per_label`
        shows and this deliberately does not hide.
        """
        return float(np.nanmean(list(self.per_label(predictions, targets).values())))
