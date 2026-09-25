"""What one candidate adds to a NAMED pool: `python -m kneemri.evaluation.seat_add --pool <stems> --candidate <stem>`.

WHY THIS EXISTS. Pricing `raptor_ft_ourtrunk44` into `honest12` required writing the vote and the macro
inline, because nothing committed does it: `seat_value` reports divergence and member axes and computes
no add at all, and `member_curve` — which does compute adds — takes `--checkpoints` and a `--device`, so
it runs models and needs the card. The holdout records already on disk are sufficient and need none. The
no-notebooks convention says a one-off run becomes an entrypoint, and this is that entrypoint (knee-fjbte).

THE ROW SET IS THE WHOLE DIFFICULTY, and it is why this reports a RELATION rather than a row count. A
marginal contribution is only a seat if the base and the base-plus-candidate answer the SAME question, so
the base is always recomputed on the studies the candidate actually covers — never taken from a table
that priced a wider frame. Three relations follow, and they are not the same finding:

  * `EXACT`    — the candidate covers the pool's shared studies and no others. Nothing is subset.
  * `NESTED`   — the candidate's studies are a SUBSET. Subsetting the pool to them is exact arithmetic,
                 so the add is exact and the only cost is a narrower question. `raptor_ft_ourtrunk44`
                 is this case: 896 studies, every one inside the pool's 909, truth byte-identical.
  * `PARTIAL`  — the candidate holds studies the pool does not. The intersection is then a third
                 question neither side was priced on, and this REFUSES rather than reporting a number.

The distinction is the point. A tool that reports only "row counts differ" turns a free, exact answer
into a blocker, which is exactly what happened for a day.

THE RULER IS LEDGER75 BY DEFAULT, not the midpoint. Every add this project quotes is a ledger75 add, and
`scores.py`'s own docstring records that the mixed cut was unreachable from any scorer-taking caller
until 2026-09-08 (knee-f9aao). `--midpoint` asks the same question under the other cut; the two are not
comparable and both are printed so neither can be quoted without its name.

WHAT IT DOES NOT DO. It does not select, rank or seat. It prices ONE candidate against ONE pool the
caller named, because every seating instrument this project has built has been closed on the board, and
a tool that offered to pick the pool would be one more of those.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from kneemri.evaluation.ledger75 import LedgerSeventyFive
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.inference.vote import LogitAverage, Vote
from kneemri.runs.holdout_record import SUFFIX, HoldoutReading, HoldoutRecord

logger = logging.getLogger(__name__)

BOARD_RESOLVABLE = 0.0015
"""The add a member must clear to be worth a seat. D174 §3's bar, measured, not chosen: zero of 242
members cleared it and the best clean add in the pool was +0.0011."""

TRUTH_TOLERANCE = 1e-6
"""How far two records' truth may differ on a shared study before they are different rulers. Records
written by the same frame agree exactly; float32 round-trips are the only slack this allows."""


class Relation(StrEnum):
    """How the candidate's studies stand to the pool's — the fact that decides whether an add exists."""

    EXACT = "EXACT"
    NESTED = "NESTED"
    PARTIAL = "PARTIAL"


@dataclass(frozen=True)
class SeatAddRow:
    """One candidate priced into one pool, with everything needed to refuse the number."""

    candidate: str
    relation: Relation
    pool_size: int
    rows: int
    base: float
    with_candidate: float
    solo: float
    ruler: str

    def add(self) -> float:
        """The marginal contribution, base recomputed on the same rows the candidate covers."""
        return self.with_candidate - self.base

    def clears(self) -> bool:
        """Whether the add clears the measured bar a seat has to beat."""
        return self.add() >= BOARD_RESOLVABLE


class SeatAdd:
    """Prices a candidate into a named pool off holdout records alone — no card, no checkpoints."""

    def __init__(self, scores: LabelScores | None = None, vote: Vote | None = None) -> None:
        """`scores` defaults to LEDGER75, the ruler every quoted add in this project was read under."""
        self.scores = scores if scores is not None else LedgerSeventyFive().scorer()
        self.vote = vote if vote is not None else LogitAverage()

    def shared(self, readings: list[HoldoutReading]) -> list[str]:
        """The studies EVERY reading covers, in a stable order so two runs of this agree."""
        common = set(readings[0].studies)
        for reading in readings[1:]:
            common &= set(reading.studies)
        return sorted(common)

    def relation(self, candidate: HoldoutReading, pool: list[str]) -> Relation:
        """Whether the candidate covers the pool's studies exactly, sits inside them, or spills outside.

        Spilling outside is the one case with no answer: the intersection is then a question neither the
        pool nor the candidate was priced on, and a number taken there would be quoted as if it were one.
        """
        studies = set(candidate.studies)
        inside = set(pool)
        if studies - inside:
            return Relation.PARTIAL
        return Relation.EXACT if studies == inside else Relation.NESTED

    def stacked(
        self, readings: list[HoldoutReading], studies: list[str]
    ) -> list[Float[NDArray[np.float64], "studies labels"]]:
        """Each reading's predictions for the named studies, joined by study NAME rather than position.

        Positions are not comparable across runs — a different fold or a dropped study reindexes the
        array while leaving its shape plausible — which is the same argument `HoldoutReading.aligned`
        makes for two members and this makes for many.
        """
        return [reading.at(studies).predictions for reading in readings]

    def truth(self, readings: list[HoldoutReading], studies: list[str]) -> Float[NDArray[np.float32], "studies labels"]:
        """The one truth table every reading agrees on, or a refusal when they do not.

        Two records disagreeing here means the members were scored by different rulers, which is the
        failure a matched delta cannot detect for itself: the arithmetic succeeds and the answer is to a
        question nobody asked.
        """
        tables = [reading.at(studies).truth for reading in readings]
        for reading, table in zip(readings[1:], tables[1:], strict=True):
            if float(np.abs(table - tables[0]).max()) > TRUTH_TOLERANCE:
                message = f"{reading.source} carries a different truth table on the shared studies - different rulers"
                raise ValueError(message)
        return tables[0]

    def macro(self, readings: list[HoldoutReading], studies: list[str]) -> float:
        """The pool's combined macro over the named studies, under this instrument's vote and cut."""
        combined = self.vote.combine(self.stacked(readings, studies))
        return self.scores.mean(combined, self.truth(readings, studies))

    def price(self, pool: list[Path], candidate: Path) -> SeatAddRow:
        """The candidate's add into the pool, base recomputed on the studies the candidate covers."""
        seated = [HoldoutRecord(path).required() for path in pool]
        contender = HoldoutRecord(candidate).required()
        relation = self.relation(contender, self.shared(seated))
        if relation is Relation.PARTIAL:
            message = (
                f"{candidate.name} holds studies the pool does not - the intersection is a third question "
                f"neither side was priced on, and an add taken there would not be a seat"
            )
            raise ValueError(message)
        rows = self.shared([*seated, contender])
        return SeatAddRow(
            candidate=candidate.name.removesuffix(SUFFIX).removesuffix(".pt"),
            relation=relation,
            pool_size=len(seated),
            rows=len(rows),
            base=self.macro(seated, rows),
            with_candidate=self.macro([*seated, contender], rows),
            solo=self.macro([contender], rows),
            ruler=self.scores.name(),
        )


def main() -> None:
    """Print one candidate's add into a named pool, with the row relation that licenses the number."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--holdouts", type=Path, required=True, help="directory of .holdout.npz records")
    command.parser.add_argument("--pool", nargs="+", required=True, help="member stems making up the base pool")
    command.parser.add_argument("--candidate", required=True, help="the member stem to price into it")
    command.parser.add_argument("--midpoint", action="store_true", help="read at the midpoint instead of LEDGER75")
    arguments = command.parsed()

    instrument = SeatAdd(LabelScores(MIDPOINT) if arguments.midpoint else None)
    row = instrument.price(
        [arguments.holdouts / f"{stem}{SUFFIX}" for stem in arguments.pool],
        arguments.holdouts / f"{arguments.candidate}{SUFFIX}",
    )
    command.report(
        [
            f"candidate {row.candidate}   ruler {row.ruler}   rows {row.rows}   relation {row.relation}",
            f"  base   k={row.pool_size:<3d} {row.base:.4f}",
            f"  +cand  k={row.pool_size + 1:<3d} {row.with_candidate:.4f}",
            f"  ADD          {row.add():+.4f}   solo {row.solo:.4f}",
            f"  {'CLEARS' if row.clears() else 'below'} the +{BOARD_RESOLVABLE:.4f} bar",
        ]
    )


if __name__ == "__main__":
    main()
