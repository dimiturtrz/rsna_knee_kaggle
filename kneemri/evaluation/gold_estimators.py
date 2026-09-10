"""The ways 696 radiologist-annotated cells can be spent on one number, and what each one costs in sd.

The gold set is 58 studies by twelve findings, and the flat macro over the findings is only one reading
of it — the one this project inherited by never asking. D297 measured the alternatives on three complete
out-of-fold systems already on disk (real arms, no simulation) and ranked them by the PAIRED difference
sd, which is the only number that decides anything at this n:

    pooled cell AUC, per-finding RANK-NORMALISED   0.0096   arm-corr 0.911   <- registered
    pair-count weighted macro                      0.0103   arm-corr 0.883
    flat macro over twelve (the ledger's)          0.0107   arm-corr 0.878
    pooled cell AUC over RAW scores                0.0085   arm-corr 0.892   <- DISQUALIFIED, not here
    inverse-variance (Hanley-McNeil)               0.0216   arm-corr 0.605   <- REJECTED, not here

TWO OF THEM ARE ABSENT ON PURPOSE, and that is most of what this module is.

The RAW pooled cell AUC is the sharpest number available and it is the wrong statistic. Pooling raw
scores across findings asks whether a positive ACL cell outranks a negative Effusion cell — a
between-finding CALIBRATION question rather than a diagnostic one. D297 priced that: per-finding logit
shifts at the size `targets.py:46-78` actually carries (sd 0.543) move it 0.0288 while leaving every
within-finding ranking bit-identical. That is 3.4x its own paired sd and wider than a label-table arm's
whole decision band — a verdict manufactured by the axis the arm is not asking about. Rank-normalising
within each finding before pooling deletes the cross-finding scale and reads 0.0000 under the same
shifts, which is the entire argument for the registered estimator.

Inverse-variance (Hanley-McNeil) weighting is rejected for a different failure. Its weights come from
each arm's OWN per-finding AUCs, so two arms of a paired comparison are weighted differently and the
pairing breaks: correlation falls 0.878 -> 0.605 and the sd doubles. Any variance-optimal weighting has
to be fixed across the arms — computed from n+ and n-, never from the areas — and once it is, it IS
pair-count weighting, which is here.

THE BOARD SCORES THE FLAT MACRO, SO EXACTLY ONE OF THESE IS BOARD-COMPARABLE. D298 read the competition's
own evaluation page — verbatim `Final Score = (1/12) Σ AUC_i`, "the macro-averaged AUC ROC". Equal weight,
twelve columns, per-column rank only. `MacroGold` is therefore the ledger's board-comparable row and the
other two are INTERNAL INSTRUMENTS: sharper on 58 studies, a different quantity from the one that decides
the competition. Use `rankpooled` to choose between two arms, never to predict a board number. And note
that pair-count weighting does not merely cost sharpness against the board — it points away from it, by
up-weighting exactly the populated columns the scorer weights identically (Fracture at three positives is
worth what Effusion is worth). The raw-pooling disqualification below now rests on that same definition
rather than on an argument: the columns never meet, so between-finding calibration is worth zero there.

A LEVEL IS NOT COMPARABLE ACROSS ESTIMATORS. The same records read 0.8927 flat-macro and 0.8645
rank-norm-pooled: 0.028 apart on ONE system, larger than the decision band it would be quoted into.
Every estimator carries a `name` for that reason, and no caller may print a level without it
(knee-f9aao).
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import override

import numpy as np
from jaxtyping import Float, Int
from numpy.typing import NDArray

from kneemri.evaluation.scores import MIDPOINT, LabelScores

RESOLVES = 2
"""Distinct labels a finding needs before an area under its curve means anything."""


@dataclass(frozen=True)
class Cells:
    """The gold cells of one family: its scores, those scores rank-normalised, and the truth.

    Both score arrays are carried because an area under a curve is invariant under any strictly
    increasing map — so a per-finding rank transform leaves every per-finding estimator bit-identical,
    and only the pooled reading can tell the two apart at all.

    THE TRANSFORM IS FIXED ON THE WHOLE SET, ONCE, and a resample subsets it rather than recomputing it.
    That is what makes the registered paired sd apply: rank-normalisation computed inside each draw would
    be a different transform per draw, which is the caveat D297 §4 attaches to its own fold table.
    """

    scores: Float[NDArray[np.float64], "studies targets"]
    ranked: Float[NDArray[np.float64], "studies targets"]
    labels: Int[NDArray[np.int64], "studies targets"]

    @classmethod
    def of(
        cls,
        scores: Float[NDArray[np.float64], "studies targets"],
        labels: Int[NDArray[np.int64], "studies targets"],
    ) -> "Cells":
        """Rank-normalise each finding onto [0, 1] so cells from different findings share one scale."""
        order = np.argsort(np.argsort(scores, axis=0), axis=0).astype(np.float64)
        return cls(scores=scores, ranked=order / max(scores.shape[0] - 1, 1), labels=labels)

    def rows(self, index: Int[NDArray[np.int64], " draw"]) -> "Cells":
        """The same cells restricted to a resample's studies, transform and all."""
        return Cells(scores=self.scores[index], ranked=self.ranked[index], labels=self.labels[index])


class GoldEstimator(ABC):
    """One way of spending 696 gold cells on a single number, and its own resolution."""

    def __init__(self) -> None:
        self.scores = LabelScores(cut=MIDPOINT)

    @property
    @abstractmethod
    def name(self) -> str:
        """What this estimator is called — never omitted beside a level it produced."""

    @abstractmethod
    def of(self, cells: Cells) -> float:
        """This estimator's reading of these cells, or NaN when nothing in them resolves."""

    def columns(self, cells: Cells) -> tuple[list[float], list[float]]:
        """Each resolving finding's area and its POSITIVE-NEGATIVE PAIR COUNT, in finding order.

        The pair count and not the positive count: an area under a curve is a rate over pairs, so its
        sampling variance falls with n+ times n-, and both terms come from the TRUTH, which the two arms
        of a paired comparison share. A weight read off either arm's own areas breaks the pairing — that
        is the Hanley rejection in this module's docstring.
        """
        areas: list[float] = []
        weights: list[float] = []
        for index in range(cells.labels.shape[1]):
            column = cells.labels[:, index]
            if len(np.unique(column)) < RESOLVES:
                continue
            areas.append(self.scores.area_under_curve(cells.scores[:, index], column))
            weights.append(float(column.sum()) * float(len(column) - column.sum()))
        return areas, weights

    def spread(self, cells: Cells, draws: Iterable[Int[NDArray[np.int64], " draw"]]) -> tuple[float, float]:
        """This estimator's level on the whole set, and the sd of that level over study resamples."""
        return self.of(cells), self._deviation(self.of(cells.rows(rows)) for rows in draws)

    def difference(
        self, cells: Cells, against: Cells, draws: Iterable[Int[NDArray[np.int64], " draw"]]
    ) -> tuple[float, float]:
        """Two families' difference under this estimator, and the sd of the DIFFERENCE.

        Paired on the studies rather than differenced after the fact: both families are read on the same
        resample, so the sampling error they share cancels — which is the whole reason a 58-study ruler
        resolves anything at all. It holds only while the two were fitted on the SAME fold assignment
        over the same studies; if the geometry drifts, the number to quote is the unpaired sd instead.
        """
        drawn = (self.of(cells.rows(rows)) - self.of(against.rows(rows)) for rows in draws)
        return self.of(cells) - self.of(against), self._deviation(drawn)

    @staticmethod
    def _deviation(drawn: Iterable[float]) -> float:
        """The sd of the draws that resolved. A draw where nothing resolved is dropped, never zeroed."""
        kept = [value for value in drawn if not np.isnan(value)]
        return float(np.std(kept)) if kept else float("nan")


class MacroGold(GoldEstimator):
    """The flat mean over the findings that resolve — what every gold number in this project has been."""

    @property
    @override
    def name(self) -> str:
        return "macro"

    @override
    def of(self, cells: Cells) -> float:
        areas, _ = self.columns(cells)
        return float(np.mean(areas)) if areas else float("nan")


class PairCountGold(GoldEstimator):
    """The same areas, weighted by each finding's pair count, so a one-positive column stops voting 1/12.

    Calibration-immune like the flat macro and 0.0004 sharper — kept because it is the change that
    cannot surprise anyone, and printed beside the registered estimator rather than instead of it.
    """

    @property
    @override
    def name(self) -> str:
        return "paircount"

    @override
    def of(self, cells: Cells) -> float:
        areas, weights = self.columns(cells)
        return float(np.average(areas, weights=weights)) if sum(weights) else float("nan")


class RankPooledGold(GoldEstimator):
    """One curve over all 696 cells at once, each finding rank-normalised first. THE REGISTERED RULER.

    Pooling is what buys the resolution — paired sd 0.0096 against the flat macro's 0.0107, at the
    highest arm-to-arm correlation measured (0.911) — and the per-finding rank transform is what makes
    pooling legitimate: it deletes the between-finding scale, so the statistic can no longer be moved by
    a recalibration that changes no ranking.
    """

    @property
    @override
    def name(self) -> str:
        return "rankpooled"

    @override
    def of(self, cells: Cells) -> float:
        flat = cells.labels.reshape(-1)
        if len(np.unique(flat)) < RESOLVES:
            return float("nan")
        return self.scores.area_under_curve(cells.ranked.reshape(-1), flat)


ESTIMATORS: tuple[GoldEstimator, ...] = (RankPooledGold(), PairCountGold(), MacroGold())
"""Declaration-only dispatch table, registered ruler first and the ledger's incumbent last.

Ordered rather than alphabetical because the first line of a report is the one that decides, and the
flat macro is printed only so a new number can be compared to every number already in the ledger."""
