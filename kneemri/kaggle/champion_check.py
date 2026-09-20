"""Can the best entry this project has still be rebuilt? `python -m kneemri.kaggle.champion_check`.

THE SEVENTH CHECK, and the first that guards a RESULT rather than a shipment. The other six ask whether
what is about to be sent is what the repo means to send. This one asks the opposite question, about
something already sent: the entry that scored 0.925 on the hidden set is the only proof this project has
that any of its choices were right, and nothing in the repo held it together. Its members live in a
roster that is edited by hand every time a new one is trained, its serving config lives in a kernel
script that gets copied and rewritten for each arm, and the number itself lived in prose.

Losing it would not announce itself. A roster line dropped while adding a member, or a `chosen` tuple
edited in place instead of copied into a new kernel, leaves everything green: the package builds, the
kit verifies, the kernel runs, and the entry that proved the recipe is simply no longer expressible.

So the champion is pinned in `comp/champion.json` and this compares that record against the two places
it actually has to hold — the roster that decides what is packaged, and the kernel that names what is
loaded. It reads the kernel with `ast` rather than a regex, because the thing being checked is what
Python would bind, and a comment holding an old member name is not a member.

WHAT IT DOES NOT DO IS FORBID CHANGE. Every entry after the champion is a deliberate deviation from it —
that is how the axes get priced — so a differing kernel is not a finding. Only the champion's OWN
reproduction path is asserted.
"""

import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from core import cli
from core.denominator import Denominator
from kneemri.kaggle.board_read import RECORD as LEADERBOARD
from kneemri.kaggle.board_read import LeaderboardRead
from kneemri.kaggle.package_members import MemberRoster

logger = logging.getLogger(__name__)

RECORD = Path("comp/champion.json")
"""The pinned best entry. Prose lives in `comp/SUBMISSIONS.md`; this is the half something can read."""

KERNELS = Path("comp/kernels")
"""Where a kernel's script lives, one directory per entry, named after the kernel."""

CHOSEN = "chosen"
"""The name a kernel binds its member tuple to, which is the only line of it this check reads."""

SHELVES = MemberRoster.shelves()
"""Every roster a kernel can mount a member from. A pin may seat members off any shelf, so a check
reading only the first called `tripleseat_v2` unbuildable over three members the second shelf ships.
It was then a literal of two, which made it wrong again the day shelf 3 opened; `shelves` discovers."""

PINNABLE = "ensemble"
"""The one scope a pin may hold. An `excluded` row is on the board and disqualified for a stated reason
(a grafted fusion, a kit failure), and a `member_mean` row is not an entry at all — neither can be a
champion however it scored, so neither may raise the staleness complaint below."""

BOARD = Path("comp/board.jsonl")
"""Every scored entry, one JSON object a line — the record `from_board` reads a NON-champion entry from.

The pin above names one entry because a champion is one thing. Pricing a member needs a roster whose
members can still be scored on the holdout, and the champion's need not be one: `allfit` scored highest
with six of its seven members fitted at `--fold -1`. So the board is the wider record, and an entry
chosen out of it carries its own id into the report's denominator.
"""


@dataclass(frozen=True)
class Champion:
    """The best scored entry, and everything needed to build it again."""

    entry: str
    kernel: str
    board: float
    members: tuple[str, ...]
    kit: str | None
    """The commit the shipped kit was built at, and None where it did not survive.

    OPTIONAL because the record is honestly incomplete and pretending otherwise costs more than the
    gap: fourteen board entries predate per-entry kernel directories, so no log holds their sha. A
    required field forces such a row to carry the string "None", which reads exactly like a commit to
    every check downstream. Nothing that PRICES an entry reads this — `seat_value` wants the members
    and the score — so demanding it would only refuse those rows the one use they are still good for.
    """

    @classmethod
    def recorded(cls, path: Path = RECORD) -> "Champion":
        """The pinned champion, read from the record rather than from whatever scored most recently."""
        held = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            entry=str(held["entry"]),
            kernel=str(held["kernel"]),
            board=float(str(held["board"])),
            members=tuple(str(name) for name in held["members"]),
            kit=str(held["kit"]),
        )

    @classmethod
    def from_board(cls, entry: str, path: Path = BOARD) -> "Champion":
        """Any scored entry, read from the board record by its id — not necessarily the best one.

        `recorded` answers "what is the champion"; this answers "price against THIS entry", which is a
        different question and became a necessary one when the champion stopped being scoreable. The
        return type is the same because the roster arithmetic downstream does not care which entry it
        was handed — only the denominator it prints does, and that is built from `entry`.
        """
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        held = {str(row["id"]): row for row in rows if "id" in row}
        if entry not in held:
            message = f"no board entry named {entry}; the record holds {', '.join(sorted(held))}"
            raise ValueError(message)
        row = held[entry]
        # A row can be present and still be unusable: 31 of 48 were written without the two fields that
        # make an entry rebuildable, and reading them straight raised a bare KeyError naming neither the
        # entry nor the field. Say which, because the repair is to go and read the kernel's `chosen`
        # tuple and its log's `kit` line for that entry.
        #
        # A NULL counts as absent, not just a missing key: a row saying `"kit": null` is DECLINING to
        # invent a sha, and a presence-only test would pass it and hand the caller `str(None)`.
        #
        # `kit` is not in this list — see the field's own note. What is required is what makes the entry
        # a comparison target at all: which kernel, what it scored, and which members it loaded.
        absent = [field for field in ("kernel", "board", "members_named") if row.get(field) is None]
        if absent:
            message = f"board entry {entry} records no {', '.join(absent)}, so nothing can be rebuilt from it"
            raise ValueError(message)
        return cls(
            entry=entry,
            kernel=str(row["kernel"]),
            board=float(str(row["board"])),
            members=tuple(str(name) for name in row["members_named"]),
            kit=str(row["kit"]) if row.get("kit") is not None else None,
        )

    def named_by_kernel(self, kernels: Path = KERNELS) -> tuple[str, ...]:
        """The members the champion's kernel binds to `chosen`, as Python would bind them.

        Parsed rather than matched: a regex reads a member name out of a comment or a docstring, and
        this check exists precisely for the case where the code and the prose around it disagree.
        """
        script = kernels / self.kernel / f"{self.kernel}.py"
        tree = ast.parse(script.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Tuple):
                continue
            if any(isinstance(target, ast.Name) and target.id == CHOSEN for target in node.targets):
                return tuple(
                    element.value
                    for element in node.value.elts
                    if isinstance(element, ast.Constant) and isinstance(element.value, str)
                )
        return ()

    def outranked(self, path: Path = BOARD) -> list[str]:
        """Scored entries the board holds ABOVE this one — the staleness `drift` is structurally blind to.

        `drift` asks whether the pin can still be rebuilt, which is a question about the past. This asks
        whether it is still the entry worth pinning, which the record itself said no check was asking:
        `comp/champion.json` carries a principle instructing a reader to update the pin in the same
        motion as reading the board, written after the pin lagged by three hours — and the pin then
        lagged again, through five submissions, with every gate green. A rule that fails twice the same
        way belongs in a check, and this one is a comparison against a file this module already reads.

        The cost of the lag is not cosmetic: `seat_value` reads its roster from the pin precisely so no
        arm header can hold a hand-typed copy, so a pin naming yesterday's entry prices every candidate
        against a champion that no longer holds the board.
        """
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [
            f"{row['id']} scored {row['board']} against this pin's {self.board}"
            for row in rows
            if row.get("scope") == PINNABLE and row.get("board") is not None and float(row["board"]) > self.board
        ]

    def drift(self, rosters: tuple[Path, ...], kernels: Path = KERNELS) -> list[str]:
        """Every way the champion can no longer be rebuilt, named. Empty means it still can."""
        named = self.named_by_kernel(kernels)
        packaged = {
            pattern.removesuffix(".pt") for roster in rosters for pattern in MemberRoster.read(roster).patterns()
        }
        found = []
        found += [
            f"{self.kernel} no longer names {name!r}, which the {self.board} entry loaded"
            for name in self.members
            if name not in named
        ]
        found += [
            f"the roster no longer packages {name!r}, so the {self.board} entry cannot be mounted"
            for name in self.members
            if name not in packaged
        ]
        return found


def main() -> None:
    """Assert the champion is still buildable: `python -m kneemri.kaggle.champion_check`."""
    command = cli.Command(Champion.__doc__, logger)
    command.parser.add_argument("--record", type=Path, default=RECORD, help="the pinned champion")
    command.parser.add_argument("--roster", type=Path, nargs="+", default=list(SHELVES), help="every shelf that ships")
    command.parser.add_argument("--kernels", type=Path, default=KERNELS, help="where kernel scripts live")
    command.parser.add_argument("--board", type=Path, default=BOARD, help="every scored entry")
    command.parser.add_argument(
        "--leaderboard", type=Path, default=LEADERBOARD, help="the last read of the public board"
    )
    arguments = command.parsed()
    champion = Champion.recorded(arguments.record)
    # A champion whose member tuple could not be parsed out of its kernel has no member to find missing,
    # so `drift` comes back empty and this check reports the entry as rebuildable over an empty roster —
    # the one thing it exists to make impossible.
    empty = f"{champion.entry} names no member — the record or its kernel did not parse"
    Denominator(len(champion.members), empty).refuse_empty()
    drift = champion.drift(tuple(arguments.roster), arguments.kernels)
    for complaint in drift:
        logger.error("%s", complaint)
    if drift:
        message = f"{champion.entry} ({champion.board}) can no longer be rebuilt"
        raise SystemExit(message)
    # A stale pin is a different defect from an unbuildable one and is reported as one: the entry is
    # intact, it is simply no longer the entry worth pinning. It still exits non-zero, because the only
    # reason this needs a check at all is that the same rule written as prose was skipped twice.
    outranked = champion.outranked(arguments.board)
    for complaint in outranked:
        logger.error("%s", complaint)
    if outranked:
        message = f"the pin lags the board: {champion.entry} ({champion.board}) is no longer the best entry"
        raise SystemExit(message)
    # The third staleness, and the one nothing could see: the pin can be rebuildable AND current and
    # still be planned against a competition position read days ago. The gap sat in this record's prose
    # at 0.008 while the board said 0.015. The live read needs credentials, so it is a separate command
    # and this only refuses its AGE — the same split board_check draws with `--live`.
    # REPORTED, never fatal. Rule 2.5 binds the WINNING submission, so a licence tag is a refusal at
    # `submit_check` — where an entry is about to be sent — and only an exposure here, where the subject
    # is an entry that has already scored. A pin that carries one is still a real result and still the
    # thing to price against; what it cannot be is the entry left standing at the end.
    exposed = [
        complaint
        for roster in arguments.roster
        for complaint in MemberRoster.read(roster).licence_refusals(set(champion.members))
    ]
    for complaint in exposed:
        logger.warning("LICENCE EXPOSURE: the pin seats %s — a clean entry that ties it replaces it", complaint)
    aged = LeaderboardRead.recorded(arguments.leaderboard).stale(LeaderboardRead.now())
    for complaint in aged:
        logger.error("%s", complaint)
    if aged:
        raise SystemExit("re-read the board: `python -m kneemri.kaggle.board_read --live`")
    logger.info(
        "%s (%s, kit %s) is still buildable: %d members named and packaged",
        champion.entry,
        champion.board,
        champion.kit,
        len(champion.members),
    )


if __name__ == "__main__":
    main()
