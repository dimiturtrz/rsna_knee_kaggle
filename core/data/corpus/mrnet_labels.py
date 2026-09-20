"""Where one MRNet exam's ACL verdict comes from — two sources, because two layouts of this corpus ship.

The Stanford download carries `train-acl.csv` and `valid-acl.csv` INSIDE the archive, beside the volumes.
The Redivis export of the same corpus does not: the 7.74 GB file downloaded on 2026-08-29 holds 3,750
arrays — 1,250 per plane, case 0000 to 1249 — and not one CSV, because that portal serves its tables from
a separate tab. Every exam here is joined THROUGH its verdict, so with no labels the pixels build nothing
at all; a second source is what makes that archive readable rather than a convenience.

Case-number parsing lives here rather than beside `MRNetExam` because both sources need it and
`mrnet_corpus` imports this module, never the reverse.
"""

import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import override

import pandas as pd

from core.data.corpus.mrnet_source import MRNetSource

logger = logging.getLogger(__name__)

ACL_LABEL = "acl"

SPLITS: tuple[str, ...] = ("train", "valid")
"""Both downloadable splits are TRAINING data here. We never score on them — the ruler is our own
fold-0 holdout — so holding 120 exams back would cost data and buy nothing."""

VALID_FIRST = 1130
"""The first case number in the validation split, and the whole of what makes a flat archive readable.

MRNet numbers its exams GLOBALLY: train is 0000-1129 and validation is 1130-1249, 1,130 + 120 = 1,250.
So an export that drops the `train/` and `valid/` directories loses no information — the split is a
function of the case number — and this constant is that function's only parameter. The withheld
120-exam test split is not numbered into this range at all, so nothing here can silently absorb it.
"""

CASE_WIDTH = 4
"""Case numbers are zero-padded to four in the archive. A label table round-tripped through a
spreadsheet or a portal export comes back as `7` where the pixels are `0007`, and the join then finds
nothing while reporting a clean 0.0% rather than an error. Both sides are padded here instead."""


class MRNetLabels(ABC):
    """One split's ACL verdicts, however they were shipped."""

    @abstractmethod
    def table(self, split: str) -> dict[str, int]:
        """Case number to 0/1 for every exam the split labels, whether or not its pixels are here."""

    @staticmethod
    def case_of(text: str) -> str | None:
        """One case number, zero-padded to the archive's width, or `None` if this is not a case at all.

        Returning `None` rather than raising is what lets a caller walk an archive whose extra members
        it does not know about — a checksum file, a `__MACOSX` stub — without enumerating them first,
        and it is also how a header row is dropped: `case` is not a numeral.
        """
        stripped = text.strip()
        return stripped.zfill(CASE_WIDTH) if stripped.isdigit() else None

    @staticmethod
    def split_of(case: str) -> str:
        """Which downloadable split a case belongs to, from its number alone. See `VALID_FIRST`."""
        return SPLITS[0] if int(case) < VALID_FIRST else SPLITS[1]

    @staticmethod
    def verdicts(table: pd.DataFrame) -> dict[str, int]:
        """A two-column case/verdict table as a dict, tolerating a header row and unpadded case numbers.

        The header is dropped by READING it rather than by declaring it — `case_of` already answers
        whether a row names an exam, and a `header=` flag would be a second, guessable answer to it.
        """
        rows = {MRNetLabels.case_of(str(case)): verdict for case, verdict in zip(table[0], table[1], strict=True)}
        return {case: int(verdict) for case, verdict in rows.items() if case is not None}


class ArchiveLabels(MRNetLabels):
    """The tables as the Stanford download ships them — `<split>-acl.csv` beside the volumes.

    Read through `MRNetSource`, which is what makes an EXTRACTED download work: the volumes already
    resolve that way, and a second reader that only understood the zip made the two halves of one
    corpus disagree about what a download is.
    """

    def __init__(self, archive: Path) -> None:
        self.archive = archive

    @override
    def table(self, split: str) -> dict[str, int]:
        """One split's ACL table, read from the headerless CSV the download ships beside the volumes."""
        wanted = f"{split}-{ACL_LABEL}.csv"
        source = MRNetSource.at(self.archive)
        for name in source.names():
            if Path(name).name == wanted:
                return self.verdicts(pd.read_csv(io.BytesIO(source.read(name)), header=None, dtype=str))
        raise FileNotFoundError(f"{wanted} is not in this download")


class TableLabels(MRNetLabels):
    """The tables fetched SEPARATELY from the pixels, for the export that ships pixels alone.

    Matched by glob rather than by exact name: an export names its files after its own table, and the one
    thing every such name carries is the split and the finding.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @override
    def table(self, split: str) -> dict[str, int]:
        """One split's ACL table, from whichever file in the directory names that split and the ACL."""
        found = sorted(self.directory.glob(f"*{split}*{ACL_LABEL}*.csv"))
        if not found:
            raise FileNotFoundError(f"no *{split}*{ACL_LABEL}*.csv under {self.directory}")
        logger.info(f"{split} labels <- {found[0]}")
        return self.verdicts(pd.read_csv(found[0], header=None, dtype=str))
