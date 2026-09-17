"""The ruler re-cut to cells the label-error detector does NOT flag: `python -m kneemri.evaluation.clean_cells`.

WHY A THIRD CUT OF THE SAME 909 STUDIES. `relabel_hold` asked whether training on a CORRECTED label table
beats training on the report labels, and its own pre-registration (`experiments/arms/relabel_hold.sh`)
named four sign patterns before it ran. It landed on the one that reopens the axis: proxy -0.0058 on the
909-study holdout, gold +0.0104 on the 58 annotated studies. The account that predicts exactly that is
RULER BIAS — our holdout labels carry the same false-positive-heavy noise as our training labels (108
false positives against 32 misses over the 58), so a member fitted on corrected labels is scored by a
ruler that still rewards the over-calling and is charged for being right.

The decline at `comp/members.declines:569` then closed the axis on POWER, not on a refutation: 58 gold
cells give a 95% half-width near 0.014 and the effects this project chases are 0.010, so the gold ruler
cannot arbitrate supervision by construction. That leaves the axis closed by a ruler that is biased and
a ruler that is blind.

WHAT THIS INSTRUMENT BUYS, AND IT COSTS NO CARD. 410 of the 1,932 corrected cells land in the 909-study
holdout; the other 10,498 cells are ones the detector does not flag, where the report label is the label
both arms were trained to agree with and the bias argument does not apply. Scored THERE, the question is
no longer contaminated and the denominator is 10,498 cells rather than 58 studies. The pre-registered
reading, which may not be re-interpreted after the number is seen:

  UP on clean cells   -> corrected supervision genuinely produces a better model, because better
                         supervision improves a model everywhere and not only on the cells it edited.
                         The ruler-bias account is confirmed and the supervision axis reopens.
  DOWN on clean cells -> the corrections are wrong rather than the ruler, and the axis closes for real.

WHAT IT CANNOT SAY. The flagged side is reported for contrast only, and per-label it is thin — Synovitis
holds 3 flagged cells in the holdout and Baker's 13, which support no per-label statement at either cut.
And both arms held their annotations out, so this is a fold-0 report-label ruler throughout: it tests
whether the corrections helped, never whether the gold labels agree.

The denominator every number here carries is "clean cells of the 909-study fold-0 holdout", and per the
cut seam (knee-f9aao) each is printed under BOTH rulers, never one alone.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool
from numpy.typing import NDArray

from core import cli
from core.findings import STUDY, TARGETS
from kneemri.evaluation.ledger75 import LedgerSeventyFive
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.runs.holdout_record import SUFFIX, HoldoutReading

logger = logging.getLogger(__name__)

TARGET_COLUMN = "target"
"""The correction table's finding column, beside `STUDY` and the soft `value` it teaches."""

PAIR = 2
"""A delta is printed only for a TREATMENT and its CONTROL — three members are a sweep, and a sweep's
best-minus-worst is a selection effect rather than a comparison anyone registered."""


@dataclass(frozen=True)
class CleanCellRow:
    """One member on both sides of the flag, under both cuts — never a side or a cut alone."""

    name: str
    clean: float
    clean75: float
    flagged: float
    flagged75: float
    clean_cells: int
    flagged_cells: int


class CleanCellRuler:
    """The 909-study holdout partitioned by whether the label-error detector flagged each (study, finding)."""

    def __init__(self, corrections: Path) -> None:
        """`corrections` is the detector's table — one row per cell it would overwrite, with a soft value."""
        with corrections.open(encoding="utf-8", newline="") as handle:
            self.flagged = frozenset((row[STUDY], row[TARGET_COLUMN]) for row in csv.DictReader(handle))

    def flagged_cells(self) -> frozenset[tuple[str, str]]:
        """Every (study, finding) the detector flagged, over the whole corpus rather than this holdout."""
        return self.flagged

    def mask(self, reading: HoldoutReading) -> Bool[NDArray[np.bool_], "s t"]:
        """True where this reading's cell is CLEAN — the selection `LabelScores.per_label` scores under.

        Built per column in the TARGETS order, because that order is the contract every prediction
        matrix in this repo is written under and a mask assembled in any other order is silently wrong
        on every finding rather than loudly wrong on one.
        """
        studies = [str(study) for study in reading.studies]
        columns = [[(study, target) not in self.flagged for study in studies] for target in TARGETS]
        return np.asarray(columns, dtype=np.bool_).T

    def per_label(self, reading: HoldoutReading, scorer: LabelScores, *, clean: bool = True) -> dict[str, float]:
        """The twelve AUCs on one side of the flag. `clean=False` reads the flagged side, for contrast."""
        selected = self.mask(reading)
        return scorer.per_label(reading.predictions, reading.truth, where=selected if clean else ~selected)

    def macro(self, reading: HoldoutReading, scorer: LabelScores, *, clean: bool = True) -> float:
        """The mean of the twelve on that side, NaN-skipping exactly as the ledger's own macro does."""
        return scorer.mean_of(self.per_label(reading, scorer, clean=clean))

    def row(self, path: Path) -> CleanCellRow:
        """One member read on both sides under both cuts, off the holdout array beside its checkpoint."""
        reading = LedgerSeventyFive().reading(path)
        recut = LedgerSeventyFive().scorer()
        midpoint = LabelScores(MIDPOINT)
        selected = self.mask(reading)
        return CleanCellRow(
            name=path.name.removesuffix(SUFFIX).removesuffix(".pt"),
            clean=self.macro(reading, midpoint),
            clean75=self.macro(reading, recut),
            flagged=self.macro(reading, midpoint, clean=False),
            flagged75=self.macro(reading, recut, clean=False),
            clean_cells=int(np.count_nonzero(selected)),
            flagged_cells=int(np.count_nonzero(~selected)),
        )

    def table(self, paths: list[Path]) -> list[CleanCellRow]:
        """Every named member, in the order the caller named them — a comparison, not a ranking.

        Unsorted deliberately: this instrument is read as a pre-registered PAIR (treatment against its
        control), and sorting by score is what turns a pair into a leaderboard nobody registered.
        """
        return [self.row(path) for path in paths]


def main() -> None:
    """Score named members on the cells the label-error detector does not flag, under both cuts."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--holdouts", type=Path, required=True, help="directory of .holdout.npz records")
    command.parser.add_argument("--corrections", type=Path, required=True, help="the detector's correction table")
    command.parser.add_argument("--members", nargs="+", required=True, help="member stems, treatment first")
    arguments = command.parsed()

    rows = CleanCellRuler(arguments.corrections).table(
        [arguments.holdouts / f"{name}{SUFFIX}" for name in arguments.members]
    )

    header = f"{'member':40s} {'CLEAN':>8s} {'CLEAN75':>9s} {'FLAGGED':>9s} {'FLAG75':>8s} {'cells':>7s} {'flag':>6s}"
    lines = [header] + [
        f"{row.name[:40]:40s} {row.clean:8.4f} {row.clean75:9.4f} "
        f"{row.flagged:9.4f} {row.flagged75:8.4f} {row.clean_cells:7d} {row.flagged_cells:6d}"
        for row in rows
    ]
    if len(rows) == PAIR:
        first, second = rows
        lines.append(
            f"{second.name} - {first.name}: CLEAN {second.clean - first.clean:+.4f} "
            f"CLEAN75 {second.clean75 - first.clean75:+.4f} over clean cells of the 909-study fold-0 holdout"
        )
    command.report(lines)


if __name__ == "__main__":
    main()
