"""Has this entry already shipped? `python -m kneemri.kaggle.submit_check --kernel <dir>`.

EVERY OTHER PRE-FLIGHT CHECK GUARDS WHAT IS IN AN ENTRY. `kit_check` guards the code, `mount_check`
the datasets, `roster_check` what may be shipped at all, `run_check` what the kernel actually
assembled, `champion_check` the pinned result. Not one of them asks whether the board already holds
this exact entry — so on 2026-08-31 a slot went to `kneemri-peak8`, seven days after `peak8_v1`
(ref 55751365, 2026-08-24, board 0.932) scored with the identical eight members. Every gate was green,
because every gate was answering a different question. See `knee-1sgr`.

SET EQUALITY ON THE MEMBERS, NEVER NAME EQUALITY ON THE KERNEL. The same tuple re-shipped under a new
kernel name is the same repeat, and renaming is exactly what happens between one attempt and the next.
Order is not part of the identity either: a vote over a set of members does not depend on the order the
tuple was typed in.

A KERNEL THAT NAMES NO MEMBERS IS UNKNOWN, and exits 2 rather than 0. Some entries rank the whole
mounted package at run time, so their seats are not in the source and this parse cannot answer. A
silent pass there would be the very defect this file exists to close — the answer "I did not look" has
to read differently from "I looked and it is new".
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from core import cli
from core.denominator import Denominator
from kneemri.kaggle.board_check import KERNELS, RECORD, BoardCensus, BoardRow
from kneemri.kaggle.kernel_script import KernelScript
from kneemri.kaggle.package_members import MemberRoster

logger = logging.getLogger(__name__)

ROSTERS = MemberRoster.shelves()
"""Where the licence tags live — EVERY shelf, not the first one. The rosters are the only files that
state, per member, which external table it trained on, so they are the only files that can answer
whether a seat is shippable at all.

This was `comp/members.roster` alone, which reads a tag on any later shelf as ABSENT — and absent is
the answer that ships. Shelf 2 carries no tag today so nothing was let through, but the gate was one
foreign member on a later shelf away from silently passing the thing it exists to refuse."""

UNKNOWN_EXIT = 2
OVERLAP_FLOOR = 0.5
"""How much of a shipped entry's seats a candidate must share before the overlap is worth a line.

Advisory only. Two ensembles drawn from a pool of a hundred members share seats constantly, and a
report that names every one of them is a report nobody reads to the end.
"""

PROSE_SHOWN = 160
"""How much of a matched row's `what` a line quotes: enough to recognise the idea, short enough to read."""


@dataclass(frozen=True)
class TriedIdea:
    """A recorded entry whose prose names every mechanism word of a candidate — the same IDEA, other seats.

    SET EQUALITY CANNOT SEE AN IDEA. `goldseat`, `twoshape` and `deepseat` each passed with a new member set
    while all three were "choose seats on a non-v2 ruler", which the board already held 0-for-3 under the
    champion (knee-b1h8y). Advisory, never a refusal: a repeated idea may still work, but the slot has to
    be spent knowing what the earlier rows concluded.
    """

    identifier: str
    board: float | None
    what: str

    @classmethod
    def matched(cls, record: Path, words: list[str]) -> list["TriedIdea"]:
        """Every recorded entry whose prose holds ALL `words`, case-folded, in the record's own order."""
        wanted = [word.casefold() for word in words]
        rows = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines() if line]
        return [
            cls(str(row["id"]), None if row.get("board") is None else float(row["board"]), str(row.get("what", "")))
            for row in rows
            if "id" in row and all(word in cls.prose(row) for word in wanted)
        ]

    @staticmethod
    def prose(row: dict[str, object]) -> str:
        """Every string a row carries, joined and case-folded — `what`, `why` and each scope note alike."""
        return " ".join(value for value in row.values() if isinstance(value, str)).casefold()

    def described(self) -> str:
        """The line a reader needs: which entry, what the board said, and the start of what it tried."""
        board = "unscored" if self.board is None else f"{self.board:.3f}"
        return f"{self.identifier} (board {board}): {self.what[:PROSE_SHOWN]}"


class SubmitCheck:
    """The board's recorded entries, asked whether a candidate kernel's seats are already one of them."""

    def __init__(self, rows: list[BoardRow]) -> None:
        self.rows = [row for row in rows if row.names_key and row.members_named]

    def repeats(self, names: set[str]) -> list[BoardRow]:
        """Every recorded entry whose seats are exactly this candidate's, newest last."""
        return [row for row in self.rows if set(row.members_named) == names]

    def overlaps(self, names: set[str]) -> list[tuple[BoardRow, float]]:
        """Recorded entries sharing most of their seats with this candidate, most-shared first.

        The fraction is of the SHIPPED entry's seats, not of the union: the question a reader has is
        "how much of that entry am I about to re-run", and a candidate that adds two seats to a shipped
        eight is a different thing from one that drops six.
        """
        shared = [(row, len(names & set(row.members_named)) / len(row.members_named)) for row in self.rows]
        near = [(row, part) for row, part in shared if OVERLAP_FLOOR <= part < 1.0]
        return sorted(near, key=lambda pair: pair[1], reverse=True)

    def described(self, row: BoardRow) -> str:
        """One recorded entry as the line a reader needs to decide whether to spend the slot."""
        board = "unscored" if row.board is None else f"{row.board:.4f}"
        return f"{row.identifier} (ref {row.ref}, {row.date}, board {board}, {len(row.members_named)} members)"

    @staticmethod
    def seats(kernel: Path) -> set[str] | None:
        """The members a kernel names plus the foreign seats it blends in, or None when its source does not say."""
        scripts = sorted(kernel.glob("*.py"))
        if not scripts:
            return None
        script = KernelScript(scripts[0])
        if script.call() is None:
            return None
        names = script.member_names()
        return set(names) | script.foreign_seats() if names else None

    @staticmethod
    def resolved(given: str, kernels: Path) -> Path:
        """A kernel named either by its directory or by its bare name, as a directory."""
        direct = Path(given)
        return direct if direct.is_dir() else kernels / given


def main() -> None:
    """Refuse a kernel whose exact member set the board already holds."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--kernel", required=True, help="the kernel directory, or its bare name")
    command.parser.add_argument("--record", type=Path, default=RECORD, help="the board record to read")
    command.parser.add_argument("--kernels", type=Path, default=KERNELS, help="where bare kernel names live")
    command.parser.add_argument("--anyway", default="", help="ship a known repeat, stating WHY in one line")
    command.parser.add_argument(
        "--roster", type=Path, nargs="+", default=list(ROSTERS), help="every shelf that ships, and its licence tags"
    )
    command.parser.add_argument(
        "--idea", action="append", default=[], help="a mechanism word of this entry; repeat it, every one must match"
    )
    arguments = command.parsed()

    kernel = SubmitCheck.resolved(arguments.kernel, arguments.kernels)
    named = SubmitCheck.seats(kernel)
    if named is None:
        logger.error("UNKNOWN %s: names no members in its source, so this check did not look", kernel.name)
        sys.exit(UNKNOWN_EXIT)

    # Before the repeat question, because a repeat is a wasted slot and this is a wasted WIN: rule 2.5
    # obliges the winner to license the submission and its source, and a member trained on a
    # no-derivatives table cannot be relicensed by us. `--anyway` does not open this one.
    refusals = [refusal for shelf in arguments.roster for refusal in MemberRoster.read(shelf).licence_refusals(named)]
    for refusal in refusals:
        logger.error("LICENCE %s: %s", kernel.name, refusal)
    if refusals:
        sys.exit(1)

    shipped = SubmitCheck(BoardCensus(record=arguments.record).rows())
    Denominator(len(shipped.rows), "no recorded entry names its members, so nothing could be compared").refuse_empty()

    for row, part in shipped.overlaps(named):
        logger.info("overlaps %.0f%% of %s", 100 * part, shipped.described(row))
    if not arguments.idea:
        logger.warning(
            "IDEA NOT CHECKED %s: pass --idea <mechanism word> to read what the board already tried", kernel.name
        )
    for tried in TriedIdea.matched(arguments.record, arguments.idea) if arguments.idea else []:
        logger.warning("TRIED %s", tried.described())
    repeats = shipped.repeats(named)
    for row in repeats:
        logger.info("SAME %d members as %s", len(named), shipped.described(row))
    if not repeats:
        command.report([f"ok {kernel.name}: {len(named)} members, no recorded entry carries this set"])
        return
    if arguments.anyway:
        logger.warning("SHIPPING A KNOWN REPEAT of %s: %s", kernel.name, arguments.anyway)
        return
    logger.error("REFUSED %s: this member set has already spent a slot. --anyway '<why>' to ship it again", kernel.name)
    sys.exit(1)


if __name__ == "__main__":
    main()
