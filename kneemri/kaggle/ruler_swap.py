"""Does the SEAT ORDER move when the ruler changes? `RulerSwap`.

`python -m kneemri.kaggle.ruler_swap --table <alternative label table> [--seats 8]`

WHY THIS EXISTS. Every seat this project has ever chosen was chosen on one ruler: the 909-study report
holdout scored against `report_labels_v2.csv`. `2026-09-05_the-holdout-is-the-distorted-ruler.md` measured
that ruler carrying the lexicon's own over-call — 108 confident false positives against 32 misses on the
gold 58 — and one set of weights reading 0.8956 on the radiologist and 0.857 on the reports. Eight board
flats followed from ordering candidates on it. If the seats are fitted to report noise, then the flats
have a named mechanism and a reseating on a cleaner ruler is the ONE reseating that is mechanism-backed
rather than a shuffle.

WHAT IT DOES. It builds the members' shared holdout once, scores the SAME predictions against two truth
tables, and runs the same greedy seat selection under each. Then it diffs the two orders. Nothing here
reads a checkpoint or touches a card: the predictions are the recorded ones and only the truth changes,
which is what makes the diff attributable to the ruler and to nothing else.

WHAT IT DOES NOT ANSWER. Whether the new table is CLEANER. That is `reader_agreement` on the 58
radiologist studies and it must be answered FIRST — a table that is merely different reorders the seats
just as happily as a table that is better, and this tool cannot tell those apart. It also says nothing
about the board: a seat order is a claim about the offline ruler, and the transfer to the board is
`board_transfer`'s question.

UNKNOWN CELLS ARE DROPPED, NOT SCORED AS ABSENT. `Targets.load` leaves a cell the reader did not read at
the 0.5 default with weight zero; binarising that at the midpoint would assert every unread finding was
NEGATIVE. So both rulers are scored through `where`, and a table that reads fewer cells is compared on
the cells it actually reads.
"""

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float

from core import cli
from core.data.labels.targets import Targets
from core.findings import TARGETS
from core.paths import CHECKPOINT_SUFFIX
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.kaggle.package_members import MemberRoster

logger = logging.getLogger(__name__)

SEATS = 8
"""How deep the greedy runs. The champion seats seven and the honest curve flattens past that, so a
disagreement below eight seats is a disagreement about what ships; past it, about nothing."""

MOVED = "MOVED"
HELD = "held"


@dataclass(frozen=True)
class Seat:
    """One greedy step: which member took the seat, and what the blend scored once it had."""

    name: str
    score: float


class RulerSwap:
    """One pool of recorded predictions, two truth tables, and the seat order each of them produces."""

    def __init__(self, models: Path, train_table: Path, members: list[str], seats: int = SEATS) -> None:
        self.models = models
        self.train_table = train_table
        self.members = members
        self.seats = seats
        self.scores = LabelScores()

    def mixed(self) -> tuple[list[str], Pool]:  # devtools-ignore: test-mirror
        """The studies every member held out, and their aligned predictions under the recorded ruler."""
        mix = RecordMix([self.models / f"{name}.pt" for name in self.members], foreign=True)
        mix.narrow()
        return mix.shared(), mix.per_member()

    def truth(
        self, table: Path, studies: list[str]
    ) -> tuple[Float[np.ndarray, "studies targets"], Bool[np.ndarray, "studies targets"]]:
        """One label table read onto THESE studies in THIS order, plus the cells it actually read.

        The mask is the weight being non-zero rather than the score being non-default, because a table
        may legitimately read a cell as 0.5 and an unread cell is only distinguishable by its weight.
        """
        targets = Targets.load(table, self.train_table, studies)
        return targets.scores, targets.weights > 0.0

    def swapped(self, pool: Pool, table: Path, studies: list[str]) -> Pool:
        """The same members and the same predictions, scored against another table's truth."""
        return replace(pool, truth=self.truth(table, studies)[0])

    def scored(
        self,
        prediction: Float[np.ndarray, "studies targets"],
        truth: Float[np.ndarray, "studies targets"],
        mask: Bool[np.ndarray, "studies targets"],
    ) -> float:
        """The competition mean over the cells this ruler read — the one number the greedy maximises."""
        return self.scores.mean_of(self.scores.per_label(prediction.astype(np.float64), truth, where=mask))

    def forward(self, pool: Pool, mask: Bool[np.ndarray, "studies targets"] | None = None) -> list[Seat]:
        """The greedy seat order under this pool's truth: at each seat, the member that adds most.

        `MemberCurve.forward` answers the same question and cannot be called here, because it scores
        every cell and this comparison is only fair on the cells a ruler read.
        """
        scored = np.ones(pool.truth.shape, dtype=np.bool_) if mask is None else mask
        taken: list[int] = []
        order: list[Seat] = []
        for _ in range(min(self.seats, len(pool))):
            best = max(
                (index for index in range(len(pool)) if index not in taken),
                key=lambda index: self.scored(pool.combined([*taken, index]), pool.truth, scored),
            )
            taken.append(best)
            order.append(Seat(pool.names[best], self.scored(pool.combined(taken), pool.truth, scored)))
        return order

    @staticmethod
    def diff(before: list[Seat], after: list[Seat]) -> list[str]:
        """Seat by seat, who sat there under each ruler — the reading the whole tool exists for."""
        lines = ["  seat   under the recorded ruler        under the candidate table"]
        for seat, (was, now) in enumerate(zip(before, after, strict=False), start=1):
            verdict = HELD if was.name == now.name else MOVED
            lines.append(f"  {seat:>4}   {was.name:<32} {now.name:<32} {verdict}")
        return lines

    @staticmethod
    def churn(before: list[Seat], after: list[Seat]) -> float:
        """The share of the seated SET that changes — the number that decides whether a reseat is real.

        Set membership rather than position: two rulers that seat the same members in a different order
        package the same entry, and the entry is what a slot buys.
        """
        seated, reseated = {seat.name for seat in before}, {seat.name for seat in after}
        return len(seated - reseated) / max(len(seated), 1)

    @staticmethod
    def roster_members(roster: Path) -> list[str]:
        """The member names a roster file declares, without its prose.

        A roster line carries a TRAILING comment as often as a leading one and spells its member WITH
        the `.pt` suffix — `res392_ws_s2026.pt  # 0.8523 solo`. Both cost a run: the comment and the
        doubled suffix each fail as a checkpoint that is not on disk rather than as a parse.
        """
        return [name.removesuffix(CHECKPOINT_SUFFIX) for name in MemberRoster.read(roster).patterns()]

    def report(self, table: Path) -> list[str]:  # devtools-ignore: test-mirror
        """Both orders, the diff, and what the churn licenses.

        BOTH greedies run over the candidate's read cells, not each over its own. A ruler that reads
        fewer cells would otherwise be compared against a denominator it does not share, and the seat
        diff would carry a change of population as well as a change of truth — two variables in the one
        number the tool exists to report.
        """
        studies, pool = self.mixed()
        candidate = self.swapped(pool, table, studies)
        mask = self.truth(table, studies)[1]
        before, after = self.forward(pool, mask), self.forward(candidate, mask)
        churn = self.churn(before, after)
        return [
            f"{len(pool)} members over {len(studies)} shared studies, {self.seats} seats deep",
            f"the candidate table reads {mask.mean():.4f} of the {len(studies) * len(TARGETS)} cells",
            *self.diff(before, after),
            f"seated-set churn {churn:.3f}: {before[-1].score:.4f} on the recorded ruler, "
            f"{after[-1].score:.4f} on the candidate",
        ]


def main() -> None:
    """Ask whether the seats this project chose are a property of its ruler."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--table", type=Path, required=True, help="the candidate label table")
    command.parser.add_argument("--roster", type=Path, default=Path("comp/members.roster"))
    command.parser.add_argument("--seats", type=int, default=SEATS)
    arguments = command.parsed()
    root = command.rooted(arguments)
    members = RulerSwap.roster_members(arguments.roster)
    swap = RulerSwap(root.models(), root.train_table(), members, seats=arguments.seats)
    command.report(swap.report(arguments.table))


if __name__ == "__main__":
    main()
