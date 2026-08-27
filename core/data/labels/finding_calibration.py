"""Does the reader's scale mean the same thing for every FINDING, or is it shifted per finding?

Run as `python -m core.data.labels.finding_calibration`.

WHAT IS ALREADY CALIBRATED, AND WHAT IS NOT. `Supervision.applied` replaces each of the reader's five
levels with the rate the radiologist was measured to agree at — `MEASURED_AGREEMENT`, one number per
level, POOLED OVER ALL TWELVE FINDINGS. That pooling is an assumption nothing has tested: it says a
0.68 on Contusion and a 0.68 on Synovitis are worth the same. `ReaderAgreement.by_label` says they are
not. Over the reader's positive calls the radiologist agrees 44.4% of the time on Contusion and 81.2%
on Synovitis, a spread of 37 points across a pooled 66.7%, and the two findings fail in OPPOSITE
directions: Contusion is over-called 20 times against 2 misses, Synovitis under-called 14 times
against 3 over-calls.

THE MODEL, WHICH IS ONE PARAMETER PER FINDING AND USES EVERY CELL. Keep the level map and add a single
logit shift per finding:

    P(truth = 1 | level v, finding f) = sigmoid( logit(m(v)) + offset(f) )

`offset` is fitted by maximum likelihood over ALL of that finding's annotated cells rather than over
its positive calls alone, so the same parameter carries both failure directions: a finding the reader
over-calls fits a negative offset, one it misses in silence fits a positive one. `m` is the existing
pooled level map, so an offset of zero reproduces today's behaviour exactly and the change is strictly
an addition to a calibration that already works.

WHY IT IS SHRUNK RATHER THAN SUBSTITUTED. Twelve findings over 684 cells is ~57 cells each, and a rate
on 57 Bernoulli draws carries a standard error big enough to manufacture a spread out of nothing. So
the per-finding offsets are drawn toward zero by an empirical-Bayes weight fitted from the data itself
— the between-finding variance is estimated by MOMENTS, subtracting the sampling variance each offset
carries from the raw spread, and a finding whose evidence is thin keeps most of the pooled value. If
the findings genuinely share one scale the estimated between-finding variance goes to zero and every
offset with it, which is the null this module can actually return.

WHAT IT COSTS AND WHAT IT CANNOT SEE. The offsets are fitted on the 58 image-read annotations, 1.3% of
the corpus — the same 684 cells `MEASURED_AGREEMENT` itself was fitted on, so this spends no
independence that the shipped calibration had not already spent. It cannot check itself: a holdout AUC
is scored against the REPORT labels, which carry the very defect these offsets correct, so the offline
ruler is biased AGAINST any table that stops agreeing with the reader. Score a member fitted this way
on the annotated read (`kneemri.evaluation.annotated_readings`) and on its lexicon binding
(`kneemri.evaluation.noise_binding`), and treat the holdout number as a floor rather than the verdict.
"""

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from core import cli
from core.data.labels.reader_agreement import MIDPOINT, ReaderAgreement
from core.data.labels.targets import Supervision
from core.findings import STUDY, TARGETS

logger = logging.getLogger(__name__)

BOUND = 8.0
"""The widest logit shift the fit will consider. sigmoid(8) is 0.9997 — a finding needing more than
this is not being calibrated, it is being replaced, and that is a different decision."""

TOLERANCE = 1e-9
PROBABILITY_FLOOR = 1e-6
"""Keeps `logit` finite on a level map that may contain an exact 0 or 1."""

POOLED = 0.0
"""The offset a finding gets when it carries no evidence of its own — by construction, today's table."""


class FindingCalibration:
    """One logit shift per finding on top of the pooled level map, shrunk by how thin its evidence is."""

    def __init__(self, supervision: Supervision | None = None) -> None:
        self.supervision = supervision or Supervision(calibrated=True)

    @staticmethod
    def valued(cells: pd.DataFrame, read: pd.DataFrame) -> pd.DataFrame:
        """Attach each cell the LEVEL the reader wrote for it, joined from the label table itself.

        Read rather than derived from the verdict token, because the token cannot answer: `YES` covers
        three distinct levels (0.68, 0.82, 0.94) whose measured reliabilities span 0.378 to 0.958, and
        a map from token to level would collapse exactly the gradation this module is fitting on.
        """
        levels = read.melt(id_vars=[STUDY], value_vars=list(TARGETS), var_name="target", value_name="value")
        levels = levels.rename(columns={STUDY: "study"})
        joined = cells.merge(levels, on=["study", "target"], how="inner")
        return joined.dropna(subset=["value"])

    def expected(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """What today's pooled calibration says each cell is worth, before any per-finding shift.

        A value the reader does not emit passes through as itself, which is what `Supervision.applied`
        does with the same table and for the same reason: a differently-scaled table must not be
        silently reinterpreted as this one.
        """
        applied = values.astype(np.float64).copy()
        for level, rate in self.supervision.measured:
            applied[np.isclose(values, level)] = rate
        return np.clip(applied, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)

    def offset(self, expected: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
        """The maximum-likelihood logit shift for one finding, by bisection on the score function.

        The Bernoulli score `sum(truth - sigmoid(logit(p) + d))` is strictly decreasing in `d`, so the
        root is unique wherever one exists and bisection cannot land on a local optimum. A finding
        whose cells are all positive or all negative has no interior root and pins at the bound, which
        the shrinkage step is what protects against — not a special case here.
        """
        base = np.log(expected / (1.0 - expected))
        observed = float(truth.sum())

        def score(shift: float) -> float:
            return observed - float((1.0 / (1.0 + np.exp(-(base + shift)))).sum())

        low, high = -BOUND, BOUND
        if score(low) < 0.0:
            return low
        if score(high) > 0.0:
            return high
        while high - low > TOLERANCE:
            middle = (low + high) / 2.0
            if score(middle) > 0.0:
                low = middle
            else:
                high = middle
        return (low + high) / 2.0

    def information(self, expected: NDArray[np.float64], shift: float) -> float:
        """Fisher information the cells carry about the shift — the inverse of its sampling variance."""
        base = np.log(expected / (1.0 - expected))
        fitted = 1.0 / (1.0 + np.exp(-(base + shift)))
        return float((fitted * (1.0 - fitted)).sum())

    def fitted(self, cells: pd.DataFrame) -> pd.DataFrame:
        """Each finding's raw offset beside the information behind it, before any shrinkage."""
        rows = []
        for target, group in cells.groupby("target"):
            expected = self.expected(group["value"].to_numpy(dtype=np.float64))
            truth = (group["truth"].to_numpy(dtype=np.float64) > MIDPOINT).astype(np.float64)
            shift = self.offset(expected, truth)
            rows.append((str(target), len(group), shift, self.information(expected, shift)))
        return pd.DataFrame(rows, columns=["target", "cells", "raw", "information"])

    def spread(self, fitted: pd.DataFrame) -> float:
        """Between-finding variance by moments: the observed spread of offsets less what sampling explains.

        Each raw offset is its own true value plus noise of variance `1 / information`, so the mean of
        the squared offsets over-states the real spread by the mean of those variances. Subtract it. A
        negative result means the findings are indistinguishable from one shared scale and the honest
        answer is zero — which returns every offset to the pooled value and this module to a no-op.
        """
        observed = float((fitted["raw"] ** 2).mean())
        sampling = float((1.0 / fitted["information"]).mean())
        return max(0.0, observed - sampling)

    def offsets(self, cells: pd.DataFrame) -> dict[str, float]:
        """The shipped table: each finding's offset drawn toward the pooled value by its own thinness.

        The weight is the standard empirical-Bayes one, `tau / (tau + variance)`: a finding with plenty
        of cells keeps nearly all of its measured shift, a thin one keeps almost none.
        """
        fitted = self.fitted(cells)
        tau = self.spread(fitted)
        if tau <= 0.0:
            return dict.fromkeys(fitted["target"], POOLED)
        weights = tau / (tau + 1.0 / fitted["information"])
        drawn = zip(fitted["target"], weights, fitted["raw"], strict=True)
        return {str(target): float(weight * raw) for target, weight, raw in drawn}

    def shifted(
        self,
        values: NDArray[np.float64],
        targets: list[str],
        offsets: dict[str, float],
    ) -> NDArray[np.float64]:
        """Every cell's calibrated value moved by its finding's offset, in logit space."""
        expected = self.expected(values)
        base = np.log(expected / (1.0 - expected))
        shifts = np.array([offsets.get(target, POOLED) for target in targets], dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-(base + shifts)))

    def report(self, cells: pd.DataFrame) -> list[str]:
        """The offsets, what they do to a mid-scale cell, and whether the spread survived its own noise."""
        fitted = self.fitted(cells)
        tau = self.spread(fitted)
        shrunk = self.offsets(cells)
        lines = [
            f"{len(cells)} annotated cells over {cells['study'].nunique()} studies, {len(fitted)} findings",
            f"between-finding variance {tau:.4f} (sd {np.sqrt(tau):.3f} in logits)",
            "  finding             cells      raw   shrunk    0.68 cell becomes",
        ]
        mid = np.array([0.68], dtype=np.float64)
        ordered = fitted.sort_values("raw")
        for target, cells_seen, raw in zip(ordered["target"], ordered["cells"], ordered["raw"], strict=True):
            name = str(target)
            moved = float(self.shifted(mid, [name], shrunk)[0])
            lines.append(f"  {name:18s} {int(cells_seen):6d} {float(raw):+8.3f} {shrunk[name]:+8.3f}    {moved:.3f}")
        pooled = float(self.expected(mid)[0])
        lines.append(f"  (pooled, today: a 0.68 cell trains at {pooled:.3f})")
        if tau <= 0.0:
            lines.append("SPREAD DID NOT SURVIVE ITS OWN SAMPLING NOISE — the findings share one scale, lever is dead")
        return lines


def main() -> None:
    """Fit and print the per-finding offsets against the annotations the pooled calibration was fitted on."""
    command = cli.Command(__doc__, logger)
    arguments = command.parsed()
    root = command.rooted(arguments)

    annotated = pd.read_csv(root.raw() / "train.csv")
    read = pd.read_csv(root.processed() / "labels" / "report_labels_v2.csv")
    cells = ReaderAgreement().cells(annotated, read)
    command.report(FindingCalibration().report(FindingCalibration.valued(cells, read)))


if __name__ == "__main__":
    main()
