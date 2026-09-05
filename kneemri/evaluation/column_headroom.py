"""What each finding's column is worth, now that the metric is known. `python -m
kneemri.evaluation.column_headroom`.

THIS MODULE ONLY BECAME MEANINGFUL ON 2026-09-04. `kneemri-colnull` submitted a constant 0.5 in every
cell of every column and read EXACTLY 0.500, which settles a question `comp/README` had carried as
unmeasured since week one: the board score is a plain unweighted mean of twelve per-column ROC AUCs,
with no offset, no prevalence weighting and no non-AUC term. So a column gaining `d` moves the board by
`d / 12`, and a per-label plan is priceable without spending a slot. Before that reading, every number
this module prints would have been arithmetic on a formula nobody had checked.

WHAT IT REPORTS. A rank-average of the member holdouts it is given, scored per finding with the
project's own `LabelScores`, sorted worst column first — and then the board value of lifting every
column that sits below a target UP to that target. The target is an argument and not a constant on
purpose: there is no principled level, so the honest form is "if the weak columns reached X, the board
moves by Y", with X named by the caller and visible in the output.

WHAT IT IS NOT. The truth here is OUR supervision binarised at its midpoint, not Kaggle truth, so the
map gives the ORDERING of weak columns under our labels and not the board's own per-column areas. The
board's own number for one column arrives separately and only one column at a time: an entry that
replaces a single column with a constant drops from the champion by exactly that column's board AUC
minus 0.5, over twelve. Until such a pair exists for a column, a lift aimed by this map is aimed by
ordering, and an arm justified by it has to say so.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from numpy.typing import NDArray

from core.findings import TARGETS
from kneemri.evaluation.scores import LabelScores
from kneemri.evaluation.stack_fit import MemberHoldout

logger = logging.getLogger(__name__)

DEFAULT_TARGET = 0.90
"""The level a lift is priced against, by default. Not a frontier number and not a claim — it is the
pool's own upper-middle, so the quoted board value is what CATCHING UP costs rather than what winning
would. Override it; it is printed with every run precisely so it cannot be quoted without its level."""


@dataclass(frozen=True)
class HeadroomRow:
    """One finding's column: its area, the positives behind it, and what reaching a target would pay."""

    finding: str
    scored: int
    positives: int
    area: float

    def shortfall(self, target: float) -> float:
        """How far this column sits below the target, floored at zero — a column already above pays nothing."""
        return max(target - self.area, 0.0)

    def line(self, target: float) -> str:
        """The printed row: the finding, its area, its shortfall in board points, and the counts."""
        return (
            f"{self.finding:<18} auc {self.area:.4f}  short {self.shortfall(target):+.4f}"
            f"  board {self.shortfall(target) / len(TARGETS):+.4f}  n={self.scored} pos={self.positives}"
        )


class ColumnHeadroom:
    """Per-finding areas over a pooled holdout, and the board value of lifting the weak ones."""

    def __init__(self, target: float = DEFAULT_TARGET) -> None:
        self.target = target

    @staticmethod
    def pooled(members: list[MemberHoldout]) -> Float[NDArray[np.float64], "studies findings"]:
        """Rank-average the members into one table, each column ranked within itself first.

        Ranked rather than averaged raw because the metric reads order and nothing else, so a member's
        probability scale is not information — and leaving it in lets one confident member dominate a
        column for a reason the score cannot see.
        """
        ranked = [np.argsort(np.argsort(member.predictions, axis=0), axis=0).astype(np.float64) for member in members]
        return np.mean(ranked, axis=0) / max(len(members[0].predictions) - 1, 1)

    def scored(
        self,
        table: Float[NDArray[np.float64], "studies findings"],
        truth: Int[NDArray[np.int64], "studies findings"],
    ) -> list[HeadroomRow]:
        """One row per finding, worst column first — the order is the whole point of the report."""
        areas = LabelScores().per_label(table, truth.astype(np.float32))
        rows = [
            HeadroomRow(
                finding=finding,
                scored=len(table),
                positives=int(truth[:, column].sum()),
                area=areas[finding],
            )
            for column, finding in enumerate(TARGETS)
        ]
        return sorted(rows, key=lambda row: row.area)

    def board_value(self, rows: list[HeadroomRow]) -> float:
        """What lifting every below-target column to the target would add to the board.

        The sum of the shortfalls over twelve, which is the metric's own arithmetic and not a model of
        it. It is an UPPER BOUND on the lane by construction: it assumes every weak column reaches the
        target and that no other column is dragged down doing it, and no mechanism here promises either.
        """
        return sum(row.shortfall(self.target) for row in rows) / len(TARGETS)


def main() -> None:
    """Pool a directory of member holdouts and report the per-column map with its board arithmetic."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=Path, required=True)
    parser.add_argument("--only", nargs="+", default=None, help="member names; every recorded member by default")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    parsed = parser.parse_args()
    names = parsed.only or [path.name.split(".")[0] for path in sorted(parsed.holdouts.glob("*.holdout.npz"))]
    members = MemberHoldout.aligned([MemberHoldout.loaded(parsed.holdouts / f"{name}.holdout.npz") for name in names])
    headroom = ColumnHeadroom(parsed.target)
    rows = headroom.scored(headroom.pooled(members), members[0].truth)
    for row in rows:
        logger.info(row.line(parsed.target))
    areas = [row.area for row in rows]
    logger.info(f"{len(members)} members, {len(members[0].predictions)} studies, mean auc {np.nanmean(areas):.4f}")
    logger.info(
        f"lifting every column below {parsed.target:.2f} up to it is worth "
        f"{headroom.board_value(rows):+.4f} of board - an UPPER BOUND, no mechanism attached"
    )


if __name__ == "__main__":
    main()
