"""Are two label tables actually separated on the gold 58? `python -m kneemri.evaluation.table_separation`.

WHY THIS EXISTS. `core.data.labels.table_blend` records that the three-reader mean reads 0.8912 against
the radiologist annotations while the best single foreign table reads 0.8927, and says in its own words
that "on 58 studies and ~696 cells that difference has no computed interval and none of these should be
quoted as separated". Every label-supervision decision this project has taken since — including a
seven-hour all-data arm dosing the blend onto six seats — has been read off that unbracketed table. A
ranking whose gaps are smaller than its own sampling error is not a ranking.

THE PAIRING IS THE POINT, AND IT IS THE SAME ARGUMENT `LabelScores.paired_interval` MAKES. Two tables
are scored on the SAME 58 studies. If a study is hard to read, it is hard for both readers, and the
DIFFERENCE does not care. Bracketing each table's absolute mean separately would quote the unpaired
error twice and conclude nothing on a corpus this small; resampling studies and recomputing BOTH means
on each draw measures what is actually left over.

THE DENOMINATOR IS 58 STUDIES, NOT 696 CELLS. The twelve per-finding AUCs are twelve readings of one
corpus, so the draw is over STUDIES — the same rule `StudyBootstrap` states and for the same reason.

WHAT A NULL HERE WOULD AND WOULD NOT SAY. An interval spanning zero says these tables are INDISTINGUISHABLE
ON THIS RULER — 58 studies is a small ruler and this is the expected outcome, not a surprise. It does not
say the tables are equivalent as SUPERVISION: a table trains on ~4,400 studies and is judged by a board
score, and a reader that is better on the training corpus can be unresolvable on 58 gold rows. It removes
the licence to prefer one over another on THIS evidence, which is the licence that has been used.
"""

import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.bootstrap import STUDY_RESAMPLES, StudyBootstrap
from kneemri.evaluation.scores import LabelScores
from kneemri.evaluation.supervision_gap import SupervisionGap

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TableSeparation:
    """Every named label table scored against the gold annotations, and each pair's paired interval."""

    gap: SupervisionGap = field(default_factory=SupervisionGap)
    scores: LabelScores = field(default_factory=LabelScores)

    def readings(
        self, train_table: Path, tables: dict[str, Path]
    ) -> tuple[Float[np.ndarray, "studies targets"], dict[str, Float[np.ndarray, "studies targets"]]]:
        """The annotations, and every table's cells put in the annotations' row order.

        One shared identifier list rather than each table finding its own rows: the pairing that makes
        this interval readable only holds if draw `i` selects the same STUDY in both arms.
        """
        identifiers, annotations = self.gap.annotated(train_table)
        return annotations, {name: self.gap.reported(path, identifiers) for name, path in tables.items()}

    def mean(
        self, values: Float[np.ndarray, "studies targets"], annotations: Float[np.ndarray, "studies targets"]
    ) -> float:
        """One table's mean AUC over the twelve findings, the number `table_blend` quotes bare."""
        return self.scores.mean_of(self.gap.against(values, annotations))

    def separation(
        self,
        first: Float[np.ndarray, "studies targets"],
        second: Float[np.ndarray, "studies targets"],
        annotations: Float[np.ndarray, "studies targets"],
        resamples: int = STUDY_RESAMPLES,
    ) -> tuple[float, tuple[float, float]]:
        """`second` minus `first` on the full sample, and that difference's 95% paired interval."""

        def delta(drawn: NDArray[np.int64]) -> float:
            truth = annotations[drawn]
            return self.mean(second[drawn], truth) - self.mean(first[drawn], truth)

        point = self.mean(second, annotations) - self.mean(first, annotations)
        return point, StudyBootstrap(resamples).interval(annotations.shape[0], delta)

    def report(
        self,
        annotations: Float[np.ndarray, "studies targets"],
        readings: dict[str, Float[np.ndarray, "studies targets"]],
        resamples: int = STUDY_RESAMPLES,
    ) -> list[str]:
        """Every table's mean, then every pair's difference with the interval that decides it."""
        lines = [f"{annotations.shape[0]} annotated studies, {len(readings)} tables, {resamples} draws"]
        lines.extend(f"{name:34s} {self.mean(values, annotations):.4f}" for name, values in readings.items())
        for left, right in combinations(readings, 2):
            point, (floor, ceiling) = self.separation(readings[left], readings[right], annotations, resamples)
            verdict = "SEPARATED" if floor > 0 or ceiling < 0 else "unresolved"
            lines.append(f"{right} - {left}: {point:+.4f}  95% [{floor:+.4f}, {ceiling:+.4f}]  {verdict}")
        return lines


def main() -> None:
    """Rank named label tables on the gold annotations and bracket every pairwise difference."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--train", type=Path, required=True, help="train.csv, which carries the gold rows")
    command.parser.add_argument("--table", action="append", required=True, metavar="NAME=PATH", help="a reader")
    command.parser.add_argument("--resamples", type=int, default=STUDY_RESAMPLES)
    arguments = command.parsed()

    tables = {entry.split("=", 1)[0]: Path(entry.split("=", 1)[1]) for entry in arguments.table}
    separation = TableSeparation()
    annotations, readings = separation.readings(arguments.train, tables)
    command.report(separation.report(annotations, readings, arguments.resamples))


if __name__ == "__main__":
    main()
