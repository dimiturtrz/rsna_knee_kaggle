"""The ledger's second ruler: `python -m kneemri.evaluation.ledger75 --holdouts <dir>`.

WHY A SECOND NUMBER AND NOT A NEW DEFAULT. Every number in the ledger was read with graded truth cut at
the midpoint, so a cut that moved globally would silently reprice the whole record. This prints the same
members under both cuts, side by side, and changes no existing semantics: `ledger_check`,
`convergence_check` and `roster_check` still read the midpoint macro they always read.

THE ARGUMENT FOR THE 0.75 CUT (researcher D25, 30 members, 896 rows against the gold-58). The label
reader's 0.68 rung means "possible", and the midpoint counts every one of those rows as a POSITIVE. On
Effusion that rung is 48% gold-positive over 27 rows, so the midpoint ruler was ranking members by how
well they found SMALL FLUID that the radiologist does not call effusion. Recut at 0.75 the LEDGER~gold
Spearman on Effusion goes -.17 -> +.74 (hold -.16 -> +.77), and the macro ordering +.59 -> +.80 (hold
+.25 -> +.71). The eight columns that already had a ruler are the control: mean |move| .18 against .47
on the four that did not.

TWO COLUMNS KEEP THE MIDPOINT, and they are not an exception bolted on to make a number look better.
Medial Meniscus (+.46 -> +.12) and Baker's (+.47 -> +.04) LOSE their ruler at 0.75, because their 0.68
rung carries real positives (n 5 and 6 of the gold set). A column's cut belongs where that column's
coin-flip rung sits, and for those two it sits below 0.75.

WHAT THIS DOES NOT FIX, so nobody reads the macro as a repair of the whole table: Synovitis keeps its
mass at the UNKNOWN 0.28 rung (41 of 57 rows, below every cut), Contusion's coin-flip rung is 0.82 which
is ABOVE every cut so a recut only makes it worse, and Fracture has no coin-flip rung at all. Those three
are a different mechanism and this ruler is silent about them.

Board evidence (researcher D27): across ~530 leaderboard pairs LEDGER75 leaves 5 inversions where the
midpoint leaves 21, and 0 of 326 at the >=.0053 floor; R^2 .80 -> .96 with one slope 1.16-1.22 where the
midpoint's wanders 1.05-1.41.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.runs.holdout_record import SUFFIX, HoldoutReading, HoldoutRecord

logger = logging.getLogger(__name__)

RECUT = 0.75
"""Where the reader's "possible" rung (0.68) stops counting as a positive. D25's measured value, not a
tuned one: it is the only cut that separates 0.68 from 0.82 given the five levels v2 actually takes."""

MIDPOINT_COLUMNS: tuple[str, ...] = ("Medial Meniscus", "Baker's")
"""The two findings whose 0.68 rung carries real positives, so the recut costs them their ruler."""


@dataclass(frozen=True)
class Ledger75Row:
    """One member under both rulers — the pair, never either number alone."""

    name: str
    ledger: float
    ledger75: float
    epoch: int
    studies: int

    def moved(self) -> float:
        """LEDGER75 minus the midpoint macro. A member's own re-pricing, not a comparison to anything."""
        return self.ledger75 - self.ledger


class LedgerSeventyFive:
    """The mixed-cut ruler: ten findings at 0.75, two at the midpoint, scored off the kept arrays."""

    def __init__(self, recut: float = RECUT, midpoint_columns: tuple[str, ...] = MIDPOINT_COLUMNS) -> None:
        self.recut = LabelScores(recut)
        self.midpoint = LabelScores(MIDPOINT)
        self.midpoint_columns = midpoint_columns

    def reading(self, path: Path) -> HoldoutReading:
        """One member's kept predictions and the truth they were scored against."""
        record = HoldoutRecord(path)
        reading = record.read() if record.exists() else None
        if reading is None:
            message = f"{path} left no holdout record - a --fold -1 member has none and can never have one"
            raise FileNotFoundError(message)
        return reading

    def per_label(self, reading: HoldoutReading) -> dict[str, float]:
        """The twelve AUCs under the mixed cut, each column read where its own coin-flip rung sits."""
        recut = self.recut.per_label(reading.predictions, reading.truth)
        midpoint = self.midpoint.per_label(reading.predictions, reading.truth)
        return {target: midpoint[target] if target in self.midpoint_columns else recut[target] for target in TARGETS}

    def macro(self, reading: HoldoutReading) -> float:
        """The mean of the twelve, NaN-skipping exactly as the ledger's own macro does."""
        return float(np.nanmean(list(self.per_label(reading).values())))

    def row(self, path: Path) -> Ledger75Row:
        """One member under both rulers, with the epoch and the row count that priced it."""
        reading = self.reading(path)
        ledger = float(np.nanmean(list(self.midpoint.per_label(reading.predictions, reading.truth).values())))
        return Ledger75Row(
            name=path.name.removesuffix(SUFFIX).removesuffix(".pt"),
            ledger=ledger,
            ledger75=self.macro(reading),
            epoch=reading.epoch,
            studies=len(reading.studies),
        )

    def table(self, paths: list[Path]) -> list[Ledger75Row]:
        """Every member that left a record, best LEDGER75 first — the order a seat decision reads.

        A member whose record is missing is SKIPPED rather than raised on, because the sweep's whole
        purpose is to price everything on disk and a dark arm sitting in the same directory must not
        take the other two hundred down with it.
        """
        rows = []
        for path in paths:
            try:
                rows.append(self.row(path))
            except FileNotFoundError:
                logger.debug("%s has no holdout record", path)
        return sorted(rows, key=lambda row: -row.ledger75)


def main() -> None:
    """Print every member on disk under the midpoint macro and LEDGER75, and the move between them."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--holdouts", type=Path, required=True, help="directory of .holdout.npz records")
    command.parser.add_argument("--only", nargs="+", default=[], help="member stems, instead of the whole directory")
    arguments = command.parsed()

    paths = (
        [arguments.holdouts / f"{name}{SUFFIX}" for name in arguments.only]
        if arguments.only
        else sorted(arguments.holdouts.glob(f"*{SUFFIX}"))
    )
    rows = LedgerSeventyFive().table(paths)

    lines = [f"{'member':40s} {'LEDGER':>8s} {'LEDGER75':>9s} {'move':>8s} {'epoch':>6s} {'rows':>6s}"]
    lines += [
        f"{row.name[:40]:40s} {row.ledger:8.4f} {row.ledger75:9.4f} {row.moved():+8.4f} {row.epoch:6d} {row.studies:6d}"
        for row in rows
    ]
    lines.append(f"{len(rows)} member(s) priced under both rulers")
    command.report(lines)


if __name__ == "__main__":
    main()
