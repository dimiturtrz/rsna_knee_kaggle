"""How many held-out studies would our ruler need to separate the members we actually choose between?

`python research/scripts/ruler_ceiling.py --holdouts D:/data/volumetric/mri/models/rsna_knee_abnormality`

WHY. The three highest members this project owns sit at offline75 0.9045 / 0.9053 / 0.9059 — a spread of
0.0014 against a paired floor of 0.0062 on 909 studies. `cut_sweep.py` ruled out the threshold as the
cause (no cut beats the incumbent, and only three distinct cuts exist), which leaves the CORPUS. This
asks what that corpus would have to be.

WHAT IT MEASURES. The paired difference between two members is bootstrapped over STUDIES at a ladder of
subsample sizes, giving the sd of that difference as a function of n. AUC is not a per-study average, so
the difference has to be recomputed on each resample rather than summed; the ladder is what turns the
result from one floor into a SCALING, and the scaling is what can be extrapolated to a corpus we do not
have.

WHAT IT DECIDES. If the n that resolves 0.0014 exceeds the whole labelled corpus, then no holdout we
could ever cut separates the top of our own line, and above ~0.904 the board is the only instrument we
own. That retires offline ranking as a way to choose a shipping member, and makes a submission slot the
scarce resource for ranking rather than for confirming.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from core.findings import TARGETS
from kneemri.evaluation.ledger75 import LedgerSeventyFive

logger = logging.getLogger(__name__)

LADDER = (200, 400, 700, 909)
"""Subsample sizes. Bounded above by the holdout itself — beyond it the scaling is extrapolated, never
measured, and the extrapolation is the whole point of fitting a slope rather than quoting one floor."""

DRAWS = 400
"""Bootstrap resamples per rung. Enough that the sd of an sd is small against the sd's own spacing."""

PAIR = ("headlr_ship_s2026", "labelwise_depth24_drop40_long_s2026")
"""The two highest members on disk, and the pair the board actually disagreed about (0.937 vs 0.941)."""

TOTAL_CORPUS = 4406
"""Every labelled study this project has (`report_labels_v2.csv`, read 2026-09-18). The ceiling on any
holdout, and therefore the ceiling on any ruler cut from one."""

SPREAD = 0.0014
"""The offline75 spread of the three members at the top of the solo line — what a ruler must resolve to
be of any use where we ship."""


def macro(ledger: LedgerSeventyFive, predictions: np.ndarray, truth: np.ndarray, rows: np.ndarray) -> float:
    """The twelve-column macro over one subset of studies, under the incumbent mixed ruler."""
    recut = ledger.recut.per_label(predictions[rows], truth[rows])
    midpoint = ledger.midpoint.per_label(predictions[rows], truth[rows])
    mixed = [midpoint[target] if target in ledger.midpoint_columns else recut[target] for target in TARGETS]
    return float(np.nanmean(mixed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ledger = LedgerSeventyFive()
    readings = [ledger.reading(next(iter(sorted(args.holdouts.rglob(f"{name}.holdout.npz"))))) for name in PAIR]
    studies = len(readings[0].truth)
    logger.info(f"{PAIR[0]} vs {PAIR[1]} over {studies} studies")

    rng = np.random.default_rng(2026)
    sizes, spreads = [], []
    for size in LADDER:
        deltas = []
        for _ in range(DRAWS):
            rows = rng.integers(0, studies, size=size)
            deltas.append(
                macro(ledger, readings[0].predictions, readings[0].truth, rows)
                - macro(ledger, readings[1].predictions, readings[1].truth, rows)
            )
        spread = float(np.std(deltas, ddof=1))
        sizes.append(size)
        spreads.append(spread)
        logger.info(f"n={size:5d}  sd(paired delta) {spread:.4f}  2 sd floor {2 * spread:.4f}")

    power = np.polyfit(np.log(sizes), np.log(spreads), 1)[0]
    logger.info(f"scaling exponent {power:+.3f} (sqrt-n law would be -0.500)")

    needed = LADDER[-1] * (2 * spreads[-1] / SPREAD) ** 2
    logger.info(f"studies needed for a 2 sd floor of {SPREAD:.4f}: {needed:,.0f} against {TOTAL_CORPUS:,} labelled")


if __name__ == "__main__":
    main()
