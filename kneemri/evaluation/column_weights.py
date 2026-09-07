"""D63's per-column member weights: the table, and the offline check that the repo path reproduces it.

`python -m kneemri.evaluation.column_weights --holdouts <dir> --table comp/colweights_v1.csv`

WHAT IT ANSWERS. The researcher fitted a per-member per-column weight matrix outside the repo and read
OOF macro +0.0028 [+0.0009, +0.0045] against the uniform block, in-sample +0.0060. Before any of that
can be shipped, the REPO's path — `ColumnWeightedLogit` plus this loader plus the frame every other
seat reading is taken over — has to produce his in-sample number from his table. A port that lands a
different number is not a smaller effect; it is a different computation, and nothing downstream of it
can be believed. So this prints the uniform macro and the weighted macro side by side, and their delta
is the quantity the equality check compares.

THE TABLE IS KEYED BY MEMBER NAME, NEVER BY ROW INDEX. A weight matrix whose rows are positional is one
roster edit away from voting a member's column with another member's weights, and nothing about that
failure looks wrong: the shape still matches, every column still sums to one, and the score moves by
about what a real seat change moves. So the CSV carries its member order in a header comment, this
loader reads it, and the caller passes the members it wants — the two are joined by name and a mismatch
refuses.

THE HONESTY LINE. 120 fitted numbers against 896 studies, fitted on the frame they are read on. The
in-sample delta this prints is NOT the effect size; it is the thing the OOF number was shrunk from
(2.1x). Quote the OOF.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.ledger75 import LedgerSeventyFive
from kneemri.evaluation.seat_swap import ANCHOR, ANCHOR_WEIGHT, GOLD_LEVELS, SeatFrame
from kneemri.inference.vote import ColumnWeightedLogit, LogitAverage
from kneemri.runs.holdout_record import UNKNOWN_EPOCH, HoldoutReading

logger = logging.getLogger(__name__)

MEMBER_HEADER = "# rows=members:"
"""How the fitted table names its own rows. A positional table is refused rather than guessed at."""


@dataclass(frozen=True)
class ColumnWeights:
    """One fitted (members x findings) mixture, and the names its rows belong to."""

    members: tuple[str, ...]
    matrix: Float[NDArray[np.float64], "members targets"]

    @classmethod
    def load(cls, path: Path) -> "ColumnWeights":
        """Read a fitted table, taking the member order from its own header comment."""
        header = [line for line in path.read_text().splitlines() if line.startswith(MEMBER_HEADER)]
        if len(header) != 1:
            message = f"{path} carries {len(header)} member-name headers; exactly one is required"
            raise ValueError(message)
        members = tuple(header[0].removeprefix(MEMBER_HEADER).strip().split(","))
        matrix = np.atleast_2d(np.loadtxt(path, delimiter=",", comments="#"))
        if matrix.shape[0] != len(members):
            message = f"{path} names {len(members)} members and holds {matrix.shape[0]} rows"
            raise ValueError(message)
        return cls(members=members, matrix=matrix)

    def shrunk(self, keep: float) -> "ColumnWeights":
        """The table pulled `1 - keep` of the way back to the uniform block it deviates from.

        D65 fitted the same 120 numbers at four shrinkages and the 896 studies cannot separate them —
        the inner-fold spread between keep=0.5 and keep=1.0 is 0.0001 to 0.0005 in every fold. Nested
        inner-CV picked 0.5 in three folds of five, and 0.5 holds the same OOF lower bound (+0.0010) at
        half the width, which is why the shipped table is a shrunk one and this is arithmetic in the
        repo rather than a second file to keep in step with the first.
        """
        uniform = np.full_like(self.matrix, 1 / len(self.members))
        return ColumnWeights(members=self.members, matrix=keep * self.matrix + (1 - keep) * uniform)

    def flattened(self, columns: tuple[str, ...]) -> "ColumnWeights":
        """Named columns given back to the uniform vote — a fit that did not hold gets no say there.

        ACL is the case: across the five outer fits the weight vectors correlate at r 0.30, the five
        disagree on who carries the column, and it reads −0.0005 OOF. A column the fit cannot reproduce
        is not a weighting, and uniform there costs nothing.
        """
        where = {name: index for index, name in enumerate(TARGETS)}
        missing = [name for name in columns if name not in where]
        if missing:
            message = f"{missing} are not findings; the columns are {TARGETS}"
            raise ValueError(message)
        matrix = self.matrix.copy()
        matrix[:, [where[name] for name in columns]] = 1 / len(self.members)
        return ColumnWeights(members=self.members, matrix=matrix)

    def written(self, path: Path) -> None:
        """Save in the same shape this class reads, so a shipped table round-trips through `load`."""
        with path.open("w") as handle:
            handle.write(f"{MEMBER_HEADER} {','.join(self.members)}\n")
            handle.write(f"# {','.join(TARGETS)}\n")
            np.savetxt(handle, self.matrix, delimiter=",")

    def vote(self, members: tuple[str, ...]) -> ColumnWeightedLogit:
        """The rule, with this table's rows reordered onto the members the caller is combining."""
        where = {name: row for row, name in enumerate(self.members)}
        missing = [name for name in members if name not in where]
        if missing:
            message = f"the table has no weights for {missing}; it was fitted over {self.members}"
            raise ValueError(message)
        return ColumnWeightedLogit(self.matrix[[where[name] for name in members]])


@dataclass(frozen=True)
class BlockFrame:
    """The block members' PROBABILITIES over one shared frame, plus the anchor's ranks and the truth.

    `SeatFrame` keeps ranks because every swap reading is taken in rank space; this rule votes in LOGIT
    space, so it needs the values before the rank transform. The frame recipe is the same one, reused
    rather than restated: intersect the members' holdouts, take truth from the anchor, drop the gold
    rows.
    """

    studies: list[str]
    truth: Float[NDArray[np.float32], "studies targets"]
    predictions: dict[str, Float[NDArray[np.float64], "studies targets"]]
    anchor: Float[NDArray[np.float64], "studies targets"]

    @classmethod
    def over(cls, holdouts: Path, members: tuple[str, ...], anchor: str = ANCHOR) -> "BlockFrame":
        """The studies every named member and the anchor held out, minus the gold rows."""
        readings = {name: SeatFrame.record(holdouts, name) for name in (*members, anchor)}
        frame = sorted(set.intersection(*[set(reading.studies) for reading in readings.values()]))
        truth = SeatFrame.taken(readings[anchor], frame, readings[anchor].truth)
        keep = ~np.all(np.isin(truth, GOLD_LEVELS), axis=1)
        studies = [study for study, kept in zip(frame, keep, strict=True) if kept]
        taken = {
            name: SeatFrame.taken(reading, frame, reading.predictions).astype(np.float64)[keep]
            for name, reading in readings.items()
        }
        return cls(
            studies=studies,
            truth=truth[keep],
            predictions={name: taken[name] for name in members},
            anchor=SeatFrame.ranked(taken[anchor]),
        )

    def macro(self, predictions: Float[NDArray[np.float64], "studies targets"]) -> float:
        """LEDGER75 over this frame — the ruler every seat reading in the ledger is taken with."""
        reading = HoldoutReading(studies=self.studies, predictions=predictions, truth=self.truth, epoch=UNKNOWN_EPOCH)
        return LedgerSeventyFive().macro(reading)

    def kernel(self, block: Float[NDArray[np.float64], "studies targets"]) -> float:
        """The champion's shape: the block's ranks at 2/3, the anchor's at 1/3."""
        mixed = (1 - ANCHOR_WEIGHT) * SeatFrame.ranked(block) + ANCHOR_WEIGHT * self.anchor
        return self.macro(mixed)

    def read(self, weights: ColumnWeights) -> tuple[float, float]:
        """(uniform, weighted) macro of the same members under the same kernel — the equality check."""
        members = tuple(self.predictions)
        stacked = [self.predictions[name] for name in members]
        return self.kernel(LogitAverage().combine(stacked)), self.kernel(weights.vote(members).combine(stacked))


def main() -> None:
    """Print the frame size and the two macros the port has to reproduce."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--holdouts", type=Path, required=True)
    command.parser.add_argument("--table", type=Path, required=True)
    command.parser.add_argument("--anchor", default=ANCHOR)
    command.parser.add_argument("--shrink", type=float, default=1.0)
    command.parser.add_argument("--uniform-column", action="append", default=[])
    command.parser.add_argument("--write", type=Path)
    command.parser.add_argument("--drop-member", action="append", default=[])
    arguments = command.parsed()

    weights = ColumnWeights.load(arguments.table).shrunk(arguments.shrink)
    weights = weights.flattened(tuple(arguments.uniform_column))
    seated = tuple(name for name in weights.members if name not in arguments.drop_member)
    frame = BlockFrame.over(arguments.holdouts, seated, arguments.anchor)
    uniform, weighted = frame.read(weights)
    if arguments.write is not None:
        weights.written(arguments.write)
    command.report(
        [
            f"frame {len(frame.studies)} studies, {len(frame.predictions)} members + {arguments.anchor}",
            f"shrink {arguments.shrink} uniform columns {arguments.uniform_column or 'none'}",
            f"uniform {uniform:.4f}  weighted {weighted:.4f}  delta {weighted - uniform:+.4f}",
        ]
    )


if __name__ == "__main__":
    main()
