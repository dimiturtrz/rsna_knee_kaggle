"""Rank the whole shelf by TAIL OVERLAP against one member: `python -m kneemri.evaluation.tail_screen
--run <stem>`.

THE AXIS EVERY EARLIER SEATING SCREEN USED WAS THE WRONG ONE. knee-jy5yu, knee-lyzc1 and knee-5hbtz each
closed the ensemble lane, and each priced candidates on SOLO score. knee-i97cx measured why that cannot
work: on `labelwise_depth24_drop40_long_s2026` the worst ninetieth of 909 holdout studies is SHARED — tail
overlap 0.631 against a chance 0.100, 8 private studies — so the members on the shelf fail the SAME
studies. A seat only pays if it fails DIFFERENT ones, which makes decorrelation the axis and solo score a
proxy that is blind to it. That is the ≤0.002 ensemble premium eightseat_v1 (8 members), swapseat_v1 (7)
and ffsolo_v1 (1) all read at board 0.944.

WHAT THIS ADDS TO `study_attribution`, WHICH ALREADY OWNS THE TAIL. `StudyAttribution.shared` answers a
different question and answers it well: it takes a SAMPLE of members and reports ONE aggregate overlap,
which is what licenses reading report and acquisition evidence off the tail. An aggregate cannot surface a
CANDIDATE — a shelf whose mean is 0.631 may still hold one member at 0.25 — so this module reuses that
module's corpus, its tail cut and its alignment rule, and changes only the reporting: every aligned member
on the shelf, ranked ascending, one row each.

WHAT A ROW MEANS, AND WHAT IT DOES NOT. Overlap is computed over the REFERENCE member's tail, each member
cut at its own tenth, so the number is "how much of this member's worst tenth does the reference already
fail". It is a pairing statistic, not a score: a member reading low may be low because it is DIFFERENT or
because it is BAD, and nothing here separates those. A candidate this finds is a candidate to PRICE, on a
board slot, against the champion — never a seat to take on this reading alone.

DECISIVE EITHER WAY. A member materially below the aggregate is a seat the solo screen could not see. A
shelf whose MINIMUM sits near the aggregate closes the ensemble lane on mechanism rather than on the
arithmetic those three beads used, which is the stronger closure and the reason this is worth the CPU.

WHAT IT READ, AND WHY THE FLOOR MAY NOT BE QUOTED ALONE. On the reference above: 391 aligned members, 51
skipped, chance 0.100, shelf mean 0.593, floor 0.022, ceiling 1.000 — and the printed verdict was
CANDIDATE, which was WRONG in the only way this module can be wrong. The bottom of the ranking decomposes
into three classes and none is a seat: WEAK (seven members, `ledger75` 0.4593 to 0.7704, the floor member
below chance AUC), VOID (the entire 0.300-0.333 band, every row a DINOv3 or raptor member reading 703 or
897 of 909 studies in-sample, `comp/members.declines:730,812,877`), and the reference member under a second
name (`solobase_hold`, overlap 1.000). The honest strong floor is 0.433, above the module's own CLOSED
threshold of 0.346, so the ensemble lane CLOSES on mechanism. Read the verdict line with a `ledger75` and an
in-sample reading beside it or not at all. The VOID band is also a finding in its own right — low overlap
at a HIGH score is the leak signature, and it separates perfectly here off two `.holdout.npz` files where
`foreign_overlap` needs the md5 split. See `interpretations/kneemri/2026-09-17_tail_overlap_closes_the_seat_lane.md`.

ALIGNMENT IS THE DENOMINATOR. `AttributionCorpus.members` keeps only members whose study list equals the
reference's exactly, so the shelf this ranks is the aligned shelf and never every checkpoint on disk. The
count of members SKIPPED is reported for that reason — a screen over four of two hundred members is not a
screen over the shelf, and the row count is the only thing that says which one happened.
"""

import logging
from dataclasses import dataclass

from core import cli
from kneemri.evaluation.study_attribution import CHANCE, AttributionCorpus, StudyAttribution
from kneemri.runs.holdout_record import SUFFIX

LOGGER = logging.getLogger(__name__)

SHELF = 512
"""How many aligned members the screen will take. Above any plausible shelf size, so it is a guard against
an unbounded read rather than a sample — a cap that BOUND the result would make the ranking alphabetical."""


@dataclass(frozen=True)
class TailOverlap:
    """One member's share of the reference member's worst tenth."""

    member: str
    overlap: float
    shared: int
    tail: int

    def line(self) -> str:
        """The row, with the count the fraction came from so a small tail cannot hide behind a ratio."""
        return f"{self.overlap:.3f}  {self.shared:3d}/{self.tail:3d}  {self.member}"


@dataclass(frozen=True)
class ShelfScreen:
    """Every aligned member ranked against one reference tail, ascending."""

    run: str
    rows: tuple[TailOverlap, ...]
    skipped: int

    def lines(self) -> list[str]:
        """The ranking, its aggregate, and the verdict the LOWEST row licenses."""
        if not self.rows:
            return [f"no member on the shelf shares {self.run}'s study order ({self.skipped} skipped)"]
        mean = sum(row.overlap for row in self.rows) / len(self.rows)
        floor = self.rows[0]
        verdict = (
            f"CANDIDATE — {floor.member} at {floor.overlap:.3f} fails studies {self.run} does not; "
            "price it on a board slot, never seat it on this reading"
            if floor.overlap < mean - (mean - CHANCE) / 2.0
            else f"CLOSED — the shelf's floor is {floor.overlap:.3f} against a mean {mean:.3f}, so no member "
            "on it fails different studies and the ensemble lane closes on mechanism"
        )
        return [
            f"{self.run}: {len(self.rows)} aligned member(s) ranked by tail overlap, {self.skipped} skipped",
            f"chance {CHANCE:.3f}   shelf mean {mean:.3f}   floor {floor.overlap:.3f}   ceiling "
            f"{self.rows[-1].overlap:.3f}",
            *[row.line() for row in self.rows],
            verdict,
        ]


class TailScreen:
    """The whole shelf priced by tail decorrelation instead of by solo score."""

    def __init__(self, corpus: AttributionCorpus, shelf: int = SHELF) -> None:
        self.corpus = corpus
        self.shelf = shelf
        self.attribution = StudyAttribution()

    def shelved(self, run: str) -> set[str]:
        """Every checkpoint on disk carrying a holdout record, minus the reference — the screen's universe."""
        models = self.corpus.models
        return {path.name.removesuffix(SUFFIX) for path in models.glob(f"*{SUFFIX}")} - {run}

    def screen(self, run: str) -> ShelfScreen:
        """Every aligned member's share of this run's tail, ascending — the lowest is the candidate."""
        rows, _table, studies, _truth = self.corpus.ranked(run)
        tail = {row.study for row in self.attribution.split(rows)[0]}
        others = self.corpus.members(run, studies, self.shelf)
        overlaps = [
            TailOverlap(
                member=name,
                overlap=len(tail & seen) / max(len(tail), 1),
                shared=len(tail & seen),
                tail=len(tail),
            )
            for name, member in others.items()
            for seen in [{row.study for row in self.attribution.split(member)[0]}]
        ]
        skipped = len(self.shelved(run)) - len(overlaps)
        return ShelfScreen(
            run=run,
            rows=tuple(sorted(overlaps, key=lambda row: (row.overlap, row.member))),
            skipped=skipped,
        )


def main() -> None:
    """Rank the shelf by how much of one member's worst tenth each other member also fails."""
    command = cli.Command(__doc__, LOGGER)
    command.parser.add_argument("--run", type=str, required=True, help="checkpoint stem whose tail is the reference")
    command.parser.add_argument("--shelf", type=int, default=SHELF, help="cap on aligned members read")
    arguments = command.parsed()
    root = command.rooted(arguments)
    corpus = AttributionCorpus(models=root.models(), raw=root.raw(), processed=root.processed())
    command.report(TailScreen(corpus, arguments.shelf).screen(arguments.run).lines())


if __name__ == "__main__":
    main()
