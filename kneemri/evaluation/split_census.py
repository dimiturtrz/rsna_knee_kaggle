"""Count the shelf by the SPLIT each member was fitted on: `python -m kneemri.evaluation.split_census`.

WHY A COUNT IS WORTH A MODULE. `tail_screen` ranked 391 aligned members and read their shared tail
failure as a fact about the recipe. It is a fact about the split: 345 of the 364 members carrying both a
checkpoint and a holdout record are fold 0 of `split_seed 2026`, so members fitted on the same 3497
studies failing the same 90 is what that construction GUARANTEES rather than evidence about
decorrelation. That correction was available from the checkpoints the whole time and nothing read it,
because no tool in this repo reports the shelf's fold composition.

WHAT IT CHANGES ABOUT THE SCREEN THAT CAME BEFORE IT. Alignment is `AttributionCorpus.members`' rule — a
member is kept only if its study order equals the reference's — and an off-fold member holds out a
DIFFERENT set by construction. So the members a tail screen reports as SKIPPED are not noise at the edge
of the reading; they are the only members whose tails could have differed. A tail screen's floor is a
statement about one fold, and this census is what says which one and how much of the shelf is in it.

WHAT IT CANNOT SAY. A fold count is provenance, not quality: the 19 off-fold members here are all the
obsolete OOF recipe at 12 slices and 8 epochs, which is why the ensemble lane closes anyway. The census
reports composition and leaves scoring to `ledger75`. And a cross-fold PAIR has no honest offline reading
on this split at all — disjoint folds hold out disjoint studies, so no study is held out by both, and
pricing such a pair needs OOF-complete predictions over the corpus instead.

UNREADABLE IS A REPORTED CLASS, NOT A SILENT DROP. A holdout record whose checkpoint is missing or will
not load carries no split stamp, so it cannot be counted — and a census that hid those would understate
its own denominator in the direction that flatters it. See
`interpretations/kneemri/2026-09-17_tail_overlap_closes_the_seat_lane.md`.
"""

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from core import cli
from kneemri.runs.holdout_record import SUFFIX

LOGGER = logging.getLogger(__name__)

OPTIONS = "options"
"""The key a checkpoint stamps its recipe under — the only part of the file this module reads."""


@dataclass(frozen=True)
class SplitCount:
    """How many members share one (fold, folds, split seed) stamp."""

    fold: str
    folds: str
    seed: str
    members: int

    def line(self) -> str:
        """The row, spelling the whole stamp so a fold number cannot be read across two splits."""
        return f"{self.members:4d}  fold {self.fold} of {self.folds}, split seed {self.seed}"


@dataclass(frozen=True)
class ShelfSplits:
    """The shelf's fold composition, commonest first."""

    rows: tuple[SplitCount, ...]
    unreadable: int

    def lines(self) -> list[str]:
        """The composition, its denominator, and what the largest stratum licenses."""
        counted = sum(row.members for row in self.rows)
        if not counted:
            return [f"no member on the shelf carries a readable split stamp ({self.unreadable} unreadable)"]
        largest = self.rows[0]
        share = largest.members / counted
        return [
            f"{counted} member(s) with a readable split stamp, {self.unreadable} unreadable",
            *[row.line() for row in self.rows],
            f"the largest stratum holds {share:.1%} of the shelf, so a screen over aligned members is a "
            f"statement about fold {largest.fold} and the members it skips are the ones that could differ",
        ]


class SplitCensus:
    """The shelf grouped by the split its members were fitted on."""

    def __init__(self, models: Path) -> None:
        self.models = models

    def stamp(self, stem: str) -> tuple[str, str, str] | None:
        """One member's (fold, folds, seed), or None when the checkpoint is missing or will not load."""
        checkpoint = self.models / f"{stem}.pt"
        if not checkpoint.exists():
            return None
        try:
            options = torch.load(checkpoint, map_location="cpu", weights_only=True).get(OPTIONS, {})
        except (RuntimeError, ValueError, EOFError):
            return None
        fold, folds, seed = (options.get(key) for key in ("fold", "folds", "split_seed"))
        if fold is None or folds is None or seed is None:
            return None
        return str(fold), str(folds), str(seed)

    def census(self) -> ShelfSplits:
        """Every member carrying a holdout record, grouped by its split stamp, commonest first."""
        counts: Counter[tuple[str, str, str]] = Counter()
        unreadable = 0
        for record in sorted(self.models.glob(f"*{SUFFIX}")):
            stamp = self.stamp(record.name.removesuffix(SUFFIX))
            if stamp is None:
                unreadable += 1
            else:
                counts[stamp] += 1
        rows = [
            SplitCount(fold=fold, folds=folds, seed=seed, members=members)
            for (fold, folds, seed), members in counts.most_common()
        ]
        return ShelfSplits(rows=tuple(rows), unreadable=unreadable)


def main() -> None:
    """Report how much of the shelf shares one training split, which is what a tail screen ranks inside."""
    command = cli.Command(__doc__, LOGGER)
    arguments = command.parsed()
    root = command.rooted(arguments)
    command.report(SplitCensus(root.models()).census().lines())


if __name__ == "__main__":
    main()
