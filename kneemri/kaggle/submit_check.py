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
from dataclasses import dataclass
from pathlib import Path

from core import cli
from core.denominator import Denominator
from kneemri.kaggle.blend_shape import Fusion
from kneemri.kaggle.board_check import KERNELS, RECORD, BoardCensus, BoardRow
from kneemri.kaggle.kernel_script import ALL_OURS, Blend, KernelScript
from kneemri.kaggle.package_members import MemberRoster

logger = logging.getLogger(__name__)

ROSTERS = MemberRoster.shelves()
"""Where the licence tags live — EVERY shelf, not the first one. The rosters are the only files that
state, per member, which external table it trained on, so they are the only files that can answer
whether a seat is shippable at all.

This was `comp/members.roster` alone, which reads a tag on any later shelf as ABSENT — and absent is
the answer that ships. Shelf 2 carries no tag today so nothing was let through, but the gate was one
foreign member on a later shelf away from silently passing the thing it exists to refuse."""

OVERLAP_FLOOR = 0.5
"""How much of a shipped entry's seats a candidate must share before the overlap is worth a line.

Advisory only. Two ensembles drawn from a pool of a hundred members share seats constantly, and a
report that names every one of them is a report nobody reads to the end.
"""

UNRECORDED = "its foreign shares and fusion are not recorded, so it counts as differing"
"""Why a foreign entry left blank in the record (per-finding shares, a nested wave) is never a match."""

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

    @classmethod
    def warnings(cls, record: Path, words: list[str], kernel: str) -> list[str]:
        """What the board already tried under these mechanism words — or that nobody asked it."""
        if not words:
            return [f"IDEA NOT CHECKED {kernel}: pass --idea <mechanism word> to read what the board already tried"]
        return [f"TRIED {tried.described()}" for tried in cls.matched(record, words)]


class SubmitCheck:
    """The board's recorded entries, asked whether a candidate kernel's seats are already one of them."""

    def __init__(self, rows: list[BoardRow], kernels: Path = KERNELS) -> None:
        self.rows = [row for row in rows if row.names_key and row.members_named]
        self.kernels = kernels

    def same_members(self, names: set[str]) -> list[BoardRow]:
        """Every recorded entry whose seats of ours are exactly this candidate's, newest last."""
        return [row for row in self.rows if set(row.members_named) == names]

    def repeats(self, names: set[str], blend: Blend) -> list[BoardRow]:
        """Every recorded entry that IS this candidate: our members, foreign shares and fusion all equal."""
        return [row for row in self.same_members(names) if self.recorded(row) == blend]

    def variants(self, names: set[str], blend: Blend) -> list[str]:
        """One line per entry sharing our members but not the blend, saying what differs. Never a refusal."""
        return [self.varied(row, blend) for row in self.same_members(names) if self.recorded(row) != blend]

    def varied(self, row: BoardRow, blend: Blend) -> str:
        """The line for one such entry: the candidate's side first, the record's second."""
        recorded = self.recorded(row)
        differs = UNRECORDED if recorded is None else ", ".join(blend.differences(recorded))
        return f"SAME members as {self.described(row)} but {differs}"

    def recorded(self, row: BoardRow) -> Blend | None:
        """The blend a row records, or None when it records none and its kernel runs a foreign seat.

        A row saying neither field is all-ours rank ONLY if nothing foreign ran: a foreign entry left blank
        because its shares were per-finding counts as differing, never as a match.
        """
        if row.foreign is not None or row.fusion is not None:
            return Blend.of(row.foreign or {}, row.fusion or Fusion.RANK)
        script = row.script(self.kernels)
        return None if script.is_file() and KernelScript(script).marked_seats() else ALL_OURS

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
        script = SubmitCheck.script(kernel)
        if script is None or script.call() is None:
            return None
        names = script.member_names()
        return set(names) | script.foreign_seats() if names else None

    def judge(self, command: cli.Command, kernel: str, entry: tuple[set[str], Blend], anyway: str) -> None:
        """Refuse an entry the record holds whole; report, never refuse, one sharing only our members."""
        named, blend = entry
        command.report(self.variants(named, blend))
        repeats = self.repeats(named, blend)
        for row in repeats:
            logger.info("SAME %d members, foreign and fusion as %s", len(named), self.described(row))
        if not repeats:
            command.report([f"ok {kernel}: {len(named)} members, no recorded entry pairs them so: {blend.described()}"])
            return
        if anyway:
            logger.warning("SHIPPING A KNOWN REPEAT of %s: %s", kernel, anyway)
            return
        logger.error("REFUSED %s: this exact entry has already spent a slot. --anyway '<why>' to ship it again", kernel)
        raise SystemExit(cli.Exit.FOUND)

    @staticmethod
    def blend(kernel: Path) -> Blend | None:
        """The kernel's foreign shares and fusion, or None when its source runs a foreign seat it does not price."""
        script = SubmitCheck.script(kernel)
        return None if script is None else script.blend()

    @staticmethod
    def script(kernel: Path) -> KernelScript | None:
        """The kernel directory's script, parsed, or None when it holds none."""
        scripts = sorted(kernel.glob("*.py"))
        return KernelScript(scripts[0]) if scripts else None

    @staticmethod
    def licence_refusals(shelves: list[Path], named: set[str]) -> list[str]:
        """Every seat any shelf tags as trained on a table whose licence forbids shipping it."""
        return [refusal for shelf in shelves for refusal in MemberRoster.read(shelf).licence_refusals(named)]

    @staticmethod
    def resolved(given: str, kernels: Path) -> Path:
        """A kernel named by its directory, its bare name or its Kaggle id `owner/name`, as a directory.

        The Kaggle id is what `kernel-metadata.json` spells, and it once resolved to `comp/kernels/<owner>/<name>`,
        a directory that never exists, which then read as a kernel naming no members (knee-z1h31).
        """
        direct = Path(given)
        return direct if direct.is_dir() else kernels / direct.name


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
    if not kernel.is_dir():
        command.parser.error(f"no kernel directory {kernel}")
    named = SubmitCheck.seats(kernel)
    if named is None:
        command.unknown(f"{kernel.name} names no members in its source, so this check did not look")

    # Before the repeat question, because a repeat is a wasted slot and this is a wasted WIN: rule 2.5
    # obliges the winner to license the submission and its source, and a member trained on a
    # no-derivatives table cannot be relicensed by us. `--anyway` does not open this one.
    refusals = SubmitCheck.licence_refusals(arguments.roster, named)
    for refusal in refusals:
        logger.error("LICENCE %s: %s", kernel.name, refusal)
    command.conclude(refusals, "licence refusal(s)")

    shipped = SubmitCheck(BoardCensus(record=arguments.record).rows(), arguments.kernels)
    Denominator(len(shipped.rows), "no recorded entry names its members, so nothing could be compared").refuse_empty()

    for row, part in shipped.overlaps(named):
        logger.info("overlaps %.0f%% of %s", 100 * part, shipped.described(row))
    for warning in TriedIdea.warnings(arguments.record, arguments.idea, kernel.name):
        logger.warning("%s", warning)
    if not shipped.same_members(named):
        command.report([f"ok {kernel.name}: {len(named)} members, no recorded entry carries this set"])
        return
    blend = SubmitCheck.blend(kernel)
    if blend is None:
        command.unknown(f"{kernel.name} runs a foreign seat whose shares its source does not state as scalars")
    shipped.judge(command, kernel.name, (named, blend), arguments.anyway)


if __name__ == "__main__":
    main()
