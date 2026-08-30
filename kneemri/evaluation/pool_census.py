"""Is there a member in this pool that is STRONG and sees the knee DIFFERENTLY — or is that cell empty?

`python -m kneemri.evaluation.pool_census --labels labels.csv`

THE QUESTION IS knee-nj4l's, AND IT IS OPEN IN ITS OWN WORDS. The champion is six copies of one lens
(every member `facebook/dinov2-small` at img=336, same six slots, same head), and seven differently
seated entries land in a 0.004 band against a 0.0053 floor. The bead says the way out is a member whose
errors are somewhere else. It also says, in a note added the day this module was written, that the
proposition has never been measured: the diverse-seating curve that appeared to refute it rested on
`mae_matched` (DINOv2-small, FOUR epochs against a 12-epoch family, peak 0.6735) and `focus_mcl` (0.5332
on the twelve-label mean, near chance on eleven columns). Those two are decorrelated because they are
UNDERTRAINED and NARROW. The pool contained no strong-and-distant member to seat, so the curve could not
have tested the proposition its conclusion was stated about.

WHAT THIS ADDS TO THE THREE RULERS IT CALLS. Nothing arithmetic — every number here comes from
`Pool.alone`, `MemberDistance` and `MemberDivergence` unchanged. What it adds is the JOIN. Each of those
answers about ONE member against ONE roster; the question above is about a POPULATION, and it is
answered by reading quality and distance on the same axis for every member on disk at once. A census is
the shape of the answer because the interesting outcome is plausibly the EMPTY one: `the cell has no
occupant across N members` is a fact that closes a P0, and it cannot be stated from any number of
one-member readings.

BOTH CUTS ARE ARGUED, NEITHER IS A ROUND NUMBER. `STRONG_OF_BEST` is this project's own measured
arm-vs-arm noise floor, so a member within it of the pool's best is one this project cannot call worse.
`distant` is defined RELATIVE to the pool — below its own median nearest-neighbour agreement — because
there is no absolute correlation that means far, and a fixed cut would be a magic number pretending to
be a threshold.

WHAT IT CANNOT SAY. Nothing here is a board reading and nothing here is a seat value. A member in the
cell is a CANDIDATE for `seat_value` to price, and `seat_value`'s own doc says the offline ruler ranks
correctly and understates what a weak member costs. The census narrows where to look; it does not
license a slot.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float

from core import cli
from kneemri.evaluation.member_distance import MemberDistance
from kneemri.evaluation.member_divergence import MemberDivergence
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import LabelScores

logger = logging.getLogger(__name__)

STRONG_OF_BEST = 0.0062
"""How far under the pool's best solo score a member may sit and still count as strong.

The project's measured arm-vs-arm noise floor, not a bar chosen here. A member inside it of the best is
one this project has no ruler fine enough to call worse, which is exactly the claim `strong` has to
carry for knee-nj4l's question: the bead is not asking for the best member, it is asking whether any
member that is not measurably weak also disagrees.
"""

CENSUSED = 3
"""The smallest pool with a median nearest-neighbour agreement worth taking.

Two members have one pairing between them, so `below the median` splits a single number against itself
and every member reads as both strong and distant. The refusal states the denominator instead.
"""

REPORTED = 12
"""How many rows of the ranked table are printed.

The census DECIDES on the cell, which is printed whole however many occupants it has; the table beneath
it is orientation — where the pool's mass sits — and past a dozen rows it is the same members in a fixed
order, which is the reason `MemberCurve` caps its own tail.
"""


@dataclass(frozen=True)
class MemberCensus:
    """One member's three coordinates: how good it is, how differently it ranks, how differently it believes."""

    name: str
    solo: float
    agreement: float
    divergence: float

    def line(self) -> str:
        """The row as it prints, in the order the question asks it: quality, then the two distances."""
        return (
            f"  {self.name}: solo {self.solo:.4f}  nearest agreement {self.agreement:.4f}  "
            f"nearest divergence {self.divergence:.4f} bits"
        )


class PoolCensus:
    """Every member on disk, on the quality axis and both distance axes at once."""

    def __init__(self, pool: Pool, scores: LabelScores) -> None:
        self.pool = pool
        self.scores = scores

    def solo(self, member: Float[np.ndarray, "studies targets"]) -> float:
        """One member's own mean score over the shared studies — the quality axis of the census."""
        return self.scores.mean(member, self.pool.truth)

    def rows(self) -> list[MemberCensus]:
        """Each member against every OTHER member: its nearest twin by rank, its nearest by belief.

        Nearest is a maximum on the agreement axis and a minimum on the divergence axis, matching the two
        rulers this delegates to: agreement rises towards a twin and divergence falls towards one. Each
        member's own seat is skipped, or every row would report the member as its own nearest neighbour.
        """
        distance, divergence = MemberDistance(self.pool, self.pool), MemberDivergence(self.pool, self.pool)
        return [
            MemberCensus(
                name,
                self.solo(member),
                max(distance.against(member, skip=index)),
                min(divergence.against(member, skip=index)),
            )
            for index, (name, member) in enumerate(zip(self.pool.names, self.pool.predictions, strict=True))
        ]

    def occupants(self, rows: list[MemberCensus]) -> list[MemberCensus]:
        """The strong-and-distant cell: within `STRONG_OF_BEST` of the best solo, and below the median agreement.

        The two cuts are read off THIS pool rather than carried in, which is what makes the empty answer
        meaningful: an absolute agreement cut could empty the cell by being drawn in the wrong place, and
        a cell emptied by its own threshold says nothing about the members.
        """
        best = max(row.solo for row in rows)
        typical = float(np.median([row.agreement for row in rows]))
        return sorted(
            (row for row in rows if row.solo >= best - STRONG_OF_BEST and row.agreement < typical),
            key=lambda row: row.agreement,
        )

    def verdict(self, rows: list[MemberCensus]) -> list[str]:
        """What the cell holds, with the denominator that makes an empty cell a finding rather than a miss."""
        best = max(row.solo for row in rows)
        typical = float(np.median([row.agreement for row in rows]))
        occupants = self.occupants(rows)
        header = (
            f"strong-and-distant cell over {len(rows)} members: solo >= {best - STRONG_OF_BEST:.4f} "
            f"(best {best:.4f} less the {STRONG_OF_BEST} floor) AND nearest agreement < {typical:.4f} "
            f"(this pool's median)"
        )
        body = (
            [
                f"  EMPTY: no member of {len(rows)} is both. knee-nj4l's open form is not refuted by this "
                f"pool, it is unrepresented in it — a seat search over these members cannot test it"
            ]
            if not occupants
            else [row.line() for row in occupants]
        )
        return [header, *body, *self.spread(rows, best)]

    def spread(self, rows: list[MemberCensus], best: float) -> list[str]:
        """How much room the distance axis has AMONG the strong, against how much it has in the pool.

        The cell above is drawn at this pool's MEDIAN agreement, which by construction leaves about half
        the pool below it — so membership is a weak reading and a count of occupants is close to
        uninformative on its own. What is threshold-free is the RANGE: if the strong members span a
        narrow band of agreement while the pool spans a wide one, then everything far is weak and
        everything strong is a twin, which is knee-nj4l's monoculture stated as a measurement instead of
        as a configuration dump.
        """
        strong = [row.agreement for row in rows if row.solo >= best - STRONG_OF_BEST]
        everyone = [row.agreement for row in rows]
        return [
            f"  nearest-agreement range among the {len(strong)} strong: {min(strong):.4f} to {max(strong):.4f}",
            f"  nearest-agreement range across all {len(everyone)}: {min(everyone):.4f} to {max(everyone):.4f}",
        ]

    def report(self) -> list[str]:
        """The cell first, because it is the decision; the ranked table after it, as orientation."""
        rows = self.rows()
        if len(rows) < CENSUSED:
            return [f"NO CENSUS: {len(rows)} members share a holdout, and a median needs at least {CENSUSED}"]
        ranked = sorted(rows, key=lambda row: row.solo, reverse=True)
        return [
            *self.verdict(rows),
            f"the pool by solo score, best first (showing {min(REPORTED, len(ranked))} of {len(ranked)}):",
            *(row.line() for row in ranked[:REPORTED]),
        ]


def main() -> None:
    """Census every member with a holdout record, on quality and both distance axes at once."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument(
        "--records",
        type=Path,
        nargs="*",
        default=None,
        help="checkpoints to census; every one with a holdout record by default",
    )
    arguments = command.parsed()

    processed = command.rooted(arguments).processed()
    records = arguments.records if arguments.records else sorted(processed.glob("*.pt"))
    mixed = RecordMix(records)
    logger.info("censusing %d checkpoints", len(records))
    for dropped in mixed.narrow():
        logger.info("%s", dropped)
    command.report(PoolCensus(mixed.per_member(), LabelScores()).report())


if __name__ == "__main__":
    main()
