"""Which cut of our own holdout tracks the BOARD best, over every solo entry we have ever shipped?

`python research/scripts/cut_sweep.py --holdouts D:/models/rsna_knee_abnormality`

WHY. `headlrsolo_v1` (2026-09-18) made three members at offline75 0.9045 / 0.9053 / 0.9059 read board
0.937 / 0.941 / 0.937. Those three are an offline TIE (spread 0.0014 against a 0.0062 paired floor), so
at the top of the range our ruler no longer separates the members we actually choose between. The
question this asks is whether that is a property of the CORPUS or of the CUT: the ledger's numbers are
midpoint, `ledger75` recuts ten of twelve columns at 0.75, and neither cut was ever chosen by how well
it predicts the board.

WHAT IT MEASURES. For each cut on a grid, every solo entry's macro AUC is recomputed from its kept
holdout arrays and regressed on that entry's board score. A cut is better if its residual scatter is
smaller: the ruler's job is to say which member will score higher, and residual sd is exactly the error
you make doing that.

WHAT IT IS NOT. Ten points and one free parameter. This is a SCREEN for a cut that is obviously better,
not a fit to adopt — a winner here owes a mechanism in the label reader's rungs (the argument that put
`ledger75` at 0.75), and a winner inside the incumbent's own scatter is no winner at all.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.findings import TARGETS
from kneemri.evaluation.ledger75 import LedgerSeventyFive
from kneemri.evaluation.scores import LabelScores

logger = logging.getLogger(__name__)

GRID = (0.50, 0.60, 0.68, 0.75, 0.80, 0.82, 0.90)
"""The cuts the label reader's five rungs can actually separate, plus the two in use."""


@dataclass(frozen=True)
class Entry:
    """One shipped solo entry: the member it served and the board it read."""

    entry: str
    member: str
    board: float


ENTRIES = (
    Entry("ortho_solo_v1", "ortho_dense_s2026", 0.907),
    Entry("resnetlong_v1", "resnet_224_long_s2026", 0.907),
    Entry("cnnsolo_v1", "resnet_224_short_s2026", 0.915),
    Entry("v4solo_v1", "labels_v4hybrid_s2026", 0.922),
    Entry("spansolo_v1", "calib_span_s2026", 0.921),
    Entry("holdsolo_v1", "calib_dense_hold_s2026", 0.928),
    Entry("matchpair_v1", "stackfit_s2026", 0.933),
    Entry("l75solo_v1", "labelwise_depth24_drop40_long_s2026", 0.941),
    Entry("extsolo_v1", "labelwise_ext_drop40_long_s2026", 0.937),
    Entry("headlrsolo_v1", "headlr_ship_s2026", 0.937),
)
"""Every single-member entry on a clean scope in `comp/board.jsonl`, read 2026-09-18."""


def fit(offline: np.ndarray, board: np.ndarray) -> tuple[float, float]:
    """Slope and residual sd of `board ~ offline`, the error of using this ruler to rank two members."""
    slope, intercept = np.polyfit(offline, board, 1)
    return float(slope), float((board - (intercept + slope * offline)).std(ddof=2))


def macro_at(scorer: LabelScores, reading: object) -> float:
    """The twelve-column macro under one flat cut."""
    per_label = scorer.per_label(reading.predictions, reading.truth)  # type: ignore[attr-defined]
    return float(np.mean([per_label[target] for target in TARGETS]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ledger = LedgerSeventyFive()
    readings = {}
    for entry in ENTRIES:
        path = args.holdouts / f"{entry.member}.holdout.npz"
        if not path.exists():
            path = next(iter(sorted(args.holdouts.rglob(f"{entry.member}.holdout.npz"))), None)
        if path is None:
            logger.info(f"MISSING {entry.entry}: no holdout arrays for {entry.member}")
            continue
        readings[entry] = ledger.reading(path)

    board = np.array([entry.board for entry in readings])
    logger.info(f"{len(readings)} of {len(ENTRIES)} entries have arrays on disk")

    mixed = np.array([ledger.macro(reading) for reading in readings.values()])
    slope, resid = fit(mixed, board)
    logger.info(f"incumbent ledger75 (mixed 0.75/midpoint): slope {slope:+.3f}  resid sd {resid:.4f}")

    for cut in GRID:
        scorer = LabelScores(cut)
        offline = np.array([macro_at(scorer, reading) for reading in readings.values()])
        slope, resid = fit(offline, board)
        logger.info(f"flat cut {cut:.2f}: slope {slope:+.3f}  resid sd {resid:.4f}  span {np.ptp(offline):.4f}")


if __name__ == "__main__":
    main()
