"""One prediction per study over the WHOLE corpus, assembled from a fold rotation.

`python -m kneemri.runs.fold_rotation --holdouts <dir> --folds <stem> <stem> ... --out <stem>`

What it prices is a RECIPE, not a checkpoint. The fold members are disposable rulers: they exist to put
one honest prediction on every study, and the thing that ships is the same recipe refit on all of them.

WHY THIS EXISTS. Every member this project owns is scored against one fixed draw — fold 0, 909 studies,
population `eb3ba199` — and the paired floor on that draw is 0.0062. A solo that beats the pinned
champion needs about +0.005 at the honest corner, which is BELOW that floor, so the ruler cannot see the
thing it is being asked to rank. The floor scales as sqrt(n): a rotation that holds out every study
exactly once prices 4407 studies instead of 909 and takes the floor to roughly 0.0028.

WHAT IT REFUSES, and why each refusal is the whole point:

  * OVERLAP. A study predicted by two fold members is a study at least one of them TRAINED on. That is
    the defect that voided `raptor_solo` at 98.6% in-sample, and it does not announce itself — the
    arithmetic succeeds and the number comes out better than it deserves. Named studies, not a count.
  * SHAPE. Records with different label counts are different rulers and cannot be stacked.
  * COVERAGE is REPORTED, never refused. A partial rotation (folds 1-3 of 5) is a legitimate thing to
    price mid-build, and a caller who assembles four of five folds is owed the number plus the fact
    that it covers 80% of the corpus — not a crash that costs them the run they already paid for.

WHAT IT DOES NOT CHECK. Across disjoint study sets there are no shared rows, so `HoldoutReading.aligned`
cannot compare the two label tables the way it does for a paired arm. Coverage against the corpus is the
strongest available substitute and it catches mis-assembly; it does NOT catch two fold members graded by
different label tables. Assemble a rotation from runs that share a table, and re-assemble after a table
moves.

The output is an ordinary `.holdout.npz`, so `ledger75`, `convergence_check` and every existing scorer
price it with no new scoring code. Its epoch is `UNKNOWN_EPOCH`: the folds kept different epochs and a
composite has no single one, and a plausible integer here would be a silent mis-join later.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from kneemri.runs.holdout_record import SUFFIX, UNKNOWN_EPOCH, HoldoutReading, HoldoutRecord

logger = logging.getLogger(__name__)

COMPOSITE_SOURCE = "out-of-fold"
"""What the assembled record says it is. A fourth value beside TRAINED/DERIVED/FOREIGN because it
differs in what a disagreement MEANS: these rows come from several sets of weights, so a per-study
residual is not attributable to one model and a witness taken on it proves nothing about any member."""

NAMED_LIMIT = 8
"""How many offending studies a refusal spells out before it counts the rest. A refusal nobody can read
is a refusal nobody acts on, and the whole list of an overlapping rotation can be thousands."""


@dataclass(frozen=True)
class FoldCoverage:
    """What an assembled rotation actually covers — the reading's own denominator, stated."""

    studies: int
    folds: int
    corpus: int

    def share(self) -> float:
        """The fraction of the corpus this rotation prices. The denominator, as a number."""
        return self.studies / self.corpus if self.corpus else 0.0

    def complete(self) -> bool:
        """Whether every corpus study got exactly one out-of-fold prediction."""
        return self.studies == self.corpus


class OutOfFoldRotation:
    """Fold records in, one corpus-wide record out. The store and its guards, not the reading."""

    def __init__(self, readings: list[HoldoutReading]) -> None:
        self._readings = readings

    def overlapping(self) -> list[str]:
        """Studies held out by more than one fold — each one an in-sample row wearing a holdout's name."""
        seen: dict[str, int] = {}
        for reading in self._readings:
            for name in reading.studies:
                seen[name] = seen.get(name, 0) + 1
        return sorted(name for name, count in seen.items() if count > 1)

    def labels(self) -> int:
        """The label count every fold must agree on, or a refusal naming the ones that do not."""
        widths = {reading.truth.shape[1] for reading in self._readings}
        if len(widths) > 1:
            message = f"fold records carry {sorted(widths)} label columns - different rulers cannot be stacked"
            raise ValueError(message)
        return widths.pop()

    def assemble(
        self, corpus: int
    ) -> tuple[
        list[str],
        Float[NDArray[np.float64], "studies labels"],
        Float[NDArray[np.float32], "studies labels"],
        FoldCoverage,
    ]:
        """Concatenate the folds into one prediction per study, refusing any study predicted twice."""
        duplicated = self.overlapping()
        if duplicated:
            shown = ", ".join(duplicated[:NAMED_LIMIT])
            rest = f" and {len(duplicated) - NAMED_LIMIT} more" if len(duplicated) > NAMED_LIMIT else ""
            message = (
                f"{len(duplicated)} study(s) are held out by more than one fold, so at least one member "
                f"TRAINED on them: {shown}{rest}"
            )
            raise ValueError(message)
        self.labels()
        studies = [name for reading in self._readings for name in reading.studies]
        predictions = np.concatenate([reading.predictions for reading in self._readings], axis=0)
        truth = np.concatenate([reading.truth for reading in self._readings], axis=0)
        coverage = FoldCoverage(studies=len(studies), folds=len(self._readings), corpus=corpus)
        return studies, predictions, truth, coverage

    def write(self, destination: Path, corpus: int) -> FoldCoverage:
        """Write the composite as an ordinary holdout record, so every existing scorer reads it."""
        studies, predictions, truth, coverage = self.assemble(corpus)
        HoldoutRecord(destination).write(studies, predictions, truth, UNKNOWN_EPOCH, source=COMPOSITE_SOURCE)
        return coverage


def main() -> None:
    """Assemble a fold rotation into one corpus-wide record and say what it covers."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--holdouts", type=Path, required=True, help="directory of .holdout.npz records")
    command.parser.add_argument("--folds", nargs="+", required=True, help="the fold members' stems, in any order")
    command.parser.add_argument("--out", required=True, help="stem for the assembled record")
    command.parser.add_argument("--corpus", type=int, required=True, help="studies the rotation should cover")
    arguments = command.parsed()

    readings = []
    for stem in arguments.folds:
        reading = HoldoutRecord(arguments.holdouts / f"{stem}{SUFFIX}").read()
        if reading is None:
            message = f"{stem} left no holdout record - a fold that predicted nothing cannot join a rotation"
            raise SystemExit(message)
        readings.append(reading)

    destination = arguments.holdouts / f"{arguments.out}{SUFFIX}"
    coverage = OutOfFoldRotation(readings).write(destination, arguments.corpus)
    command.report(
        [
            f"{coverage.folds} fold(s) -> {coverage.studies} studies, "
            f"{coverage.share():.1%} of {coverage.corpus}"
            f"{'' if coverage.complete() else '  PARTIAL - the denominator is the share above, not the corpus'}",
            f"wrote {destination}",
        ]
    )


if __name__ == "__main__":
    main()
