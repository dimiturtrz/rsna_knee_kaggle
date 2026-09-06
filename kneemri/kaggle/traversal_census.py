"""Did the memo fire? `python -m kneemri.kaggle.traversal_census --studies 96`.

`decode_bench --union` counts what the champion's three read groups WOULD open, analytically, from
`band_indices` and the file list: 204.9 opens a study against 105.6 distinct. That is a property of the
arithmetic, not of the code that ships — a traversal that passes the equality test while still opening
204.9 files is the memo not firing, and the union count would say nothing about it.

This asks the shipped path instead. It decodes every study twice through `Submission`:

- CHUNK-OUTER, the traversal that ships — every read of a study in one `decoded_reads` call against one
  `FileMemo`, which is where the between-group overlap becomes intra-call and collectable.
- GROUP-OUTER, what it replaced — one memo per read, so nothing is shared across groups. Same code, same
  reader, only the memo's SCOPE differs, which is the whole difference between the two traversals.

and reports both bills off the memos' own counters plus the thing that licenses the change at all:
whether the two paths hand every member byte-identical bags. `test_sharing` proves that for one
synthetic series; this proves it for the corpus, which is the denominator that matters.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from core import cli
from core.data.corpus.decode_bench import CHAMPION_GROUPS
from core.data.corpus.slot_reader import Band, FileMemo, Reading
from core.data.corpus.slots import SlotScheme
from kneemri.inference.ensemble import Read
from kneemri.kaggle.submission import Submission

logger = logging.getLogger(__name__)

STUDIES = 96
"""The same corpus slice `decode_bench` prices, so the two numbers are about one population."""

SIZE = 336
"""The grid the champion's seats are served on. It changes no count here — `SlotReader.cropped` opens
its files before any resize — and is named rather than defaulted so the read is the shipped one."""


@dataclass(frozen=True)
class CensusRow:
    """One study's two bills and whether the two traversals agreed to the byte."""

    slices: int
    slices_apart: int
    headers: int
    headers_apart: int
    matched: bool


class TraversalCensus:
    """Counts what each traversal opens for the champion's read groups, and whether they agree."""

    @staticmethod
    def reads() -> list[Read]:
        """The champion's three read groups as `Read`s, off `decode_bench`'s recorded tuple.

        Imported rather than respelled: the groups are a fact about `raptorweight_v1`'s recipes, and a
        second copy is a second thing to update when a seat changes band.
        """
        return [
            Read(
                slices=slices,
                scheme=SlotScheme(),
                sizes=(SIZE,),
                reading=Reading(band=Band(low=low, high=high)),
            )
            for slices, (low, high) in CHAMPION_GROUPS
        ]

    @staticmethod
    def equal(left: list[dict[int, tuple[Tensor, Tensor]]], right: list[dict[int, tuple[Tensor, Tensor]]]) -> bool:
        """Byte equality of every bag of every read — the memo's admission ticket.

        Both the pixels and the presence mask, because a memo that dropped a slot would leave the pixels
        of the slots it kept untouched and change only the mask.
        """
        if len(left) != len(right):
            return False
        return all(
            one.keys() == other.keys()
            and all(
                torch.equal(one[size][0], other[size][0]) and torch.equal(one[size][1], other[size][1]) for size in one
            )
            for one, other in zip(left, right, strict=True)
        )

    @staticmethod
    def census(directory: Path, planes: dict[str, str]) -> CensusRow:
        """One study decoded both ways, counted off the memos rather than off the arithmetic.

        An unreadable study is not skipped: `decoded_at` contributes zeros under both traversals, which
        agree, and the counters stay at whatever the failed decode reached. That is the honest reading —
        a study that opens nothing costs nothing under either traversal.
        """
        reads = TraversalCensus.reads()
        shared = FileMemo()
        together = Submission.decoded_reads(directory, planes, reads, shared)
        separate = [FileMemo() for _ in reads]
        apart = [
            Submission.decoded_at(directory, planes, Submission.pipeline(read, memo), read)
            for read, memo in zip(reads, separate, strict=True)
        ]
        return CensusRow(
            slices=shared.slice_opens,
            slices_apart=sum(memo.slice_opens for memo in separate),
            headers=shared.header_opens,
            headers_apart=sum(memo.header_opens for memo in separate),
            matched=TraversalCensus.equal(together, apart),
        )

    @staticmethod
    def report(rows: list[CensusRow]) -> list[str]:
        """Per-study bills for both traversals, and the equality count that licenses the swap."""
        studies = max(len(rows), 1)
        slices = sum(row.slices for row in rows)
        slices_apart = sum(row.slices_apart for row in rows)
        headers = sum(row.headers for row in rows)
        headers_apart = sum(row.headers_apart for row in rows)
        matched = sum(1 for row in rows if row.matched)
        return [
            f"{len(rows)} studies, {len(CHAMPION_GROUPS)} read groups",
            f"slice decodes/study  group-outer {slices_apart / studies:6.1f}   "
            f"chunk-outer {slices / studies:6.1f}   removed {1 - slices / max(slices_apart, 1):.1%}",
            f"header parses/study  group-outer {headers_apart / studies:6.1f}   "
            f"chunk-outer {headers / studies:6.1f}   removed {1 - headers / max(headers_apart, 1):.1%}",
            f"byte-identical bags: {matched} of {len(rows)} studies",
        ]


def main() -> None:
    """Decode the corpus slice under both traversals and print what the memo removed."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--studies", type=int, default=STUDIES)
    arguments = command.parsed()

    from core.data.corpus.study_cache import CorpusSplit  # noqa: PLC0415 - CLI-only

    split = CorpusSplit(command.rooted(arguments).raw())
    planes = split.planes()
    studies = [path for path in sorted(split.source.iterdir()) if path.is_dir()][: arguments.studies]
    command.report(TraversalCensus.report([TraversalCensus.census(study, planes) for study in studies]))


if __name__ == "__main__":
    main()
