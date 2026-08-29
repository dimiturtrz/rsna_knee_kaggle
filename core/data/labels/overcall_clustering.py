"""Is the reader's over-call a property of the STUDY, or twelve independent coin flips?

`python -m core.data.labels.overcall_clustering`

THE QUESTION THIS DECIDES, AND WHY IT DECIDES IT BEFORE ANY CARD IS SPENT. `noise_binding` measured the
label lane LIVE: every member tracks the reader's confident false positives at +0.54 to +0.70, and
`--calibrated-targets` is the only axis that has ever moved it (-0.145 and -0.150 at two independent
corners). The natural next mechanism after `finding_calibration` retired the per-FINDING version is a
per-STUDY one — shrink a study's targets where that study's reader pass is likely to have over-called.

A per-study weight can only work if there IS a per-study latent. If each cell's over-call were an
independent draw at the pooled rate, the per-study counts would be Binomial and no amount of per-study
shrinkage could find anything; the weight would fit sampling noise exactly as twelve per-finding scalars
did. So the cheap question is not "which covariate predicts over-call" but "is the over-call
over-dispersed relative to independent cells at all", and that needs no covariate, no card and no slot.

WHY NOT A TEXT COVARIATE, WHICH IS THE OBVIOUS ALTERNATIVE. `ReaderAgreement.signal_report` already
priced the two report-text flags and both came back BACKWARDS, with the cause named: a regex cannot
supply an independent weight because the reader has already read the same sentence, and its five verdict
levels ARE its confidence in that text. Dispersion asks the reader's OUTPUT PATTERN instead of its input,
which is the one signal it did not already consume.

WHAT THE NUMBER IS. The variance of the per-study over-call count divided by the variance a Binomial with
the same per-study cell count and the pooled rate would have. `1.0` is independence. Above `1.0` is
clustering, and the excess is what a per-study weight would have to grip.

THE DENOMINATOR, STATED HERE BECAUSE IT IS SMALL. The unit is the STUDY, and there are 56 of them once
the annotated frame is joined — not the 58 the annotation count is usually quoted as, and not the 311
positive calls those studies carry. So the dispersion ratio is resolved to roughly its own bootstrap
width and nothing finer. It can tell a strong clustering from none. It cannot rank two clustering
strengths, and no arm should be designed off its magnitude — only off which side of `1.0` the interval
falls on.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.data.labels.reader_agreement import DEFAULT_TABLE, MIDPOINT, POSITIVE, ReaderAgreement

logger = logging.getLogger(__name__)

DRAWS = 2000
"""Bootstrap resamples of the STUDIES, which is the unit the clustering is claimed at.

Resampling cells would destroy the very structure being measured — a bootstrap has to leave the
hypothesis intact to have an interval about it.
"""

INTERVAL = (2.5, 97.5)
"""The percentiles reported, matching `ReaderAgreement`'s so two readings can sit in one report."""

SEED = 2026
"""The project's split seed, reused so the interval is reproducible rather than fresh each run."""

INDEPENDENT = 1.0
"""The dispersion of cells that do not know which study they are in — the null this reads against."""


class OvercallClustering:
    """Per-study over-call counts, and whether they are wider than independence allows."""

    def __init__(self, cells: pd.DataFrame) -> None:
        self.cells = cells

    def counts(self) -> pd.DataFrame:
        """One row per study: how many cells the reader called positive, and how many of those were wrong.

        The denominator is the reader's POSITIVE calls and not the study's cells. An over-call is only
        available where a YES was said, so dividing by all twelve findings would mix the reader's
        willingness to speak into a measurement of whether it was right when it did.
        """
        positives = self.cells[self.cells["verdict"] == POSITIVE].copy()
        positives["overcalled"] = (positives["truth"] <= MIDPOINT).astype(float)
        grouped = positives.groupby("study")["overcalled"]
        return pd.DataFrame({"called": grouped.size(), "overcalled": grouped.sum()})

    @staticmethod
    def dispersion(counts: pd.DataFrame) -> float:
        """Observed variance of the per-study over-call count over the Binomial variance at the pooled rate.

        The expected variance is summed per study rather than taken at the mean cell count, because the
        studies do not all carry the same number of positive calls and a Binomial's variance is not
        linear in its own count.
        """
        rate = counts["overcalled"].sum() / counts["called"].sum()
        expected = float((counts["called"] * rate * (1.0 - rate)).mean())
        observed = float(counts["overcalled"].to_numpy(float).var())
        return observed / expected if expected else float("nan")

    def resampled(self, counts: pd.DataFrame) -> Float[NDArray[np.float64], " draws"]:
        """The dispersion under `DRAWS` resamples of the studies, the unit the claim is made at."""
        generator = np.random.default_rng(SEED)
        studies = counts.index.to_numpy()
        drawn = generator.integers(0, len(studies), size=(DRAWS, len(studies)))
        return np.asarray([self.dispersion(counts.iloc[row]) for row in drawn], dtype=np.float64)

    def report(self) -> list[str]:
        """The reading, and the branch it fires — written so the decision is not left to the reader."""
        counts = self.counts()
        ratio = self.dispersion(counts)
        low, high = np.percentile(self.resampled(counts), INTERVAL)
        rate = counts["overcalled"].sum() / counts["called"].sum()
        lines = [
            f"{len(counts)} annotated studies, {int(counts['called'].sum())} positive calls, "
            f"{int(counts['overcalled'].sum())} over-called ({rate:.1%})",
            f"per-study over-call dispersion {ratio:.2f} [{low:.2f}, {high:.2f}] against {INDEPENDENT:.2f} "
            f"for independent cells",
        ]
        if low > INDEPENDENT:
            lines.append(
                "CLUSTERED: the over-call is a property of the study, so a per-study target weight has "
                "something to grip. This licenses the arm; it does not price it."
            )
        elif high < INDEPENDENT:
            lines.append(
                "UNDER-DISPERSED: the reader over-calls more evenly than chance across studies. A "
                "per-study weight has nothing to find, and the direction is the opposite of the one "
                "proposed — suspect the census before designing anything on it."
            )
        else:
            lines.append(
                "INDEPENDENT within this interval: the per-study latent a shrinkage would fit is not "
                "measurable on this many studies. Do NOT spend card on a per-study weight; the lane's "
                "remaining headroom is not in the per-study levels either."
            )
        return lines


def main() -> None:
    """Report whether the reader's over-call clusters by study, on the annotated studies alone."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, help="the reader's table; defaults to report_labels_v2.csv")
    arguments = command.parsed()

    root = command.rooted(arguments)
    read = pd.read_csv(arguments.labels or root.processed() / "labels" / DEFAULT_TABLE)
    agreement = ReaderAgreement()
    cells = agreement.cells(agreement.annotated(root), read)

    command.report(OvercallClustering(cells).report())


if __name__ == "__main__":
    main()
