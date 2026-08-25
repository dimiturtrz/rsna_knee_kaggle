"""What are we actually behind? `python -m kneemri.kaggle.board_read --live`.

THE GAP TO THE LEADER IS THE ONLY NUMBER HERE THAT NOTHING COULD READ. `champion_check` asks whether the
best entry can be rebuilt, `board_check` whether the submission record is complete, `seat_value` what a
candidate is worth against an entry we own — every one of them is about US. The competition's own
position was reachable only by remembering to type the Kaggle CLI by hand, so it lived in prose: the
champion's principles pinned the gap at 0.008, it was 0.015 when read from the API on 2026-08-24, and
nothing noticed the decay because nothing was looking.

That is the same failure this repo has already paid for twice with `board_transfer`, whose slope was
quoted from prose at 4.67 and then 2.14 while the tool read 1.92. A number that lives in a document and
not in a command goes stale, and then gets planned against.

So: the live read is one injected round trip, the result is written to `comp/leaderboard.json` with the
UTC instant it was taken, and `champion_check` refuses a read too old to plan against. The staleness
bound is not a taste: five submissions a day reset at 00:00 UTC, so a read from before yesterday is a
number taken ahead of ten slots that have since been spent, ours and theirs alike.

THE GAP WAS ITSELF THE WRONG NUMBER, and this module printed it for a day before anything noticed. The
first version asked the API for the board's first page — twenty rows, best-first — and reported
`leader − ours`. On 2026-08-25 that read `leader 0.952, ours 0.938, gap 0.014`, which every note since
has quoted as though one team stood between us and the top. The full board, downloaded the same minute,
puts us **48th of 2,373**: forty-seven teams ahead, and forty-two of them inside 0.936–0.940. A gap of
0.014 across a plateau that thick is not one step, it is the whole field, and a plan made against
`gap` cannot see that because a page of twenty structurally cannot contain us.

So the read is now the WHOLE board (`competition_leaderboard_download`, still no slot), and what it
reports is our RANK and the size of the band we are tied inside. The gap survives as a property because
it is still the right number for "how much score is left", but it is no longer the headline, and
`ours` is now read off the board itself rather than trusted from `champion.json` — two sources that
should agree, and a disagreement worth hearing about.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core import cli

logger = logging.getLogger(__name__)

COMPETITION = "rsna-knee-abnormality-detection"
"""The slug. A WRONG one answers 403 rather than 404, which reads as a permissions problem."""

RECORD = Path("comp/leaderboard.json")
"""Where a read is kept, so the next reader gets a number and a date rather than a number."""

USERNAME = "dimiturnt"
"""Our account as the BOARD names it — an identity, not a credential, and nothing here reads a key.

Matched against `TeamMemberUserNames` rather than `TeamName`, because a team name is display text the
owner may change at any time while the username is the join key Kaggle itself uses.
"""

STALE_DAYS = 2
"""How old a read may be and still be worth planning against.

Derived, not chosen: the competition grants five submissions a day and the counter resets at 00:00 UTC,
so a read taken before yesterday sits behind at least ten spent slots — ours and every rival's. Two days
is the point past which the recorded gap describes a board that no longer exists.
"""


@dataclass(frozen=True)
class BoardRow:
    """One team's line on the public board, as the downloaded leaderboard states it."""

    rank: int
    score: float
    members: str
    """The comma-joined Kaggle usernames of that team, which is how we find OUR row without
    depending on a team NAME the owner is free to change between reads."""

    def ours(self, username: str) -> bool:
        """Whether this row is the one we are standing on."""
        return username in self.members.split(",")


@dataclass(frozen=True)
class Standing:
    """Where we sit on the board — the number a plan is actually made against, and it is not the gap."""

    rank: int
    score: float
    field: int
    tied: int
    """How many teams share our score to the board's own precision, us included.

    This is the number that makes a gap readable. Being 0.014 behind means one thing when the board
    thins out above us and quite another when forty-two teams sit inside five thousandths — the first
    is a rival to pass and the second is a plateau where a board increment buys twenty places.
    """

    @classmethod
    def of(cls, username: str, rows: list[BoardRow]) -> "Standing":
        """Our line, found by USERNAME. Raises where we are not on the board — absent is not last."""
        mine = [row for row in rows if row.ours(username)]
        if not mine:
            message = f"no row on the public board lists {username}, so no standing can be read"
            raise ValueError(message)
        row = mine[0]
        return cls(
            rank=row.rank,
            score=row.score,
            field=len(rows),
            tied=sum(1 for other in rows if other.score == row.score),
        )


@dataclass(frozen=True)
class LeaderboardRead:
    """One look at the public board: what the leader holds, where we stand, and when it was taken."""

    leader: float
    standing: Standing
    taken: str
    """The UTC instant of the round trip, ISO-8601. Passed in rather than clocked inside, so the record
    a test writes is the record it asserts on."""

    @staticmethod
    def now() -> str:
        """The clock, in one place, so every caller stamps the same shape and no test reaches for its own."""
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @property
    def ours(self) -> float:
        """Our score as the BOARD states it, not as our own record claims it."""
        return self.standing.score

    @property
    def gap(self) -> float:
        """How much score is left between us and the top — real, and no longer the headline."""
        return self.leader - self.ours

    @classmethod
    def taken_live(
        cls,
        now: str,
        username: str = USERNAME,
        viewer: Callable[[], list[BoardRow]] | None = None,
    ) -> "LeaderboardRead":
        """Ask Kaggle for the whole public board, or ask whatever was injected in its place.

        Injected for the reason `board_check.LiveBoard` and `mount_check.LiveReading` give: a path
        reachable only through real credentials is a path with no test.
        """
        rows = viewer() if viewer is not None else cls._kaggle()
        if not rows:
            message = f"the leaderboard for {COMPETITION} came back empty, so no standing can be read"
            raise ValueError(message)
        return cls(leader=max(row.score for row in rows), standing=Standing.of(username, rows), taken=now)

    @staticmethod
    def _kaggle() -> list[BoardRow]:
        """The real round trip, kept in one place so every other path is credential-free.

        The DOWNLOAD, not the view: `competition_leaderboard_view` returns the board's first page and a
        team outside the top twenty is structurally absent from it, which is exactly the blindness this
        module was built with. Neither call spends a submission slot.

        Decoded `utf-8-sig`: the CSV Kaggle ships carries a BOM, so under plain `utf-8` the first column
        is named `﻿Rank` and every row raises `KeyError: 'Rank'` — a header that LOOKS right in a log.
        """
        import csv  # noqa: PLC0415
        import io  # noqa: PLC0415
        import zipfile  # noqa: PLC0415

        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

        api = KaggleApi()
        api.authenticate()
        into = RECORD.parent
        api.competition_leaderboard_download(COMPETITION, str(into))
        archive = max(into.glob(f"{COMPETITION}*.zip"), key=lambda path: path.stat().st_mtime)
        try:
            with zipfile.ZipFile(archive) as held:
                text = held.read(held.namelist()[0]).decode("utf-8-sig")
        finally:
            archive.unlink()
        return [
            BoardRow(
                rank=int(line["Rank"]),
                score=float(line["Score"]),
                members=str(line["TeamMemberUserNames"]),
            )
            for line in csv.DictReader(io.StringIO(text))
        ]

    def written(self, path: Path = RECORD) -> Path:
        """Put the read where the next reader will find it without credentials."""
        held = {
            "why": "The public board as of `taken`, written by kneemri.kaggle.board_read. Never edit by "
            "hand and never quote from prose — re-read it, the standing decays every day slots are "
            "spent. RANK is the headline, not gap: a gap across a crowded plateau is not one step.",
            "leader": self.leader,
            "ours": self.ours,
            "gap": round(self.gap, 4),
            "rank": self.standing.rank,
            "field": self.standing.field,
            "tied": self.standing.tied,
            "taken": self.taken,
        }
        path.write_text(json.dumps(held, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def recorded(cls, path: Path = RECORD) -> "LeaderboardRead":
        """The last read, from the record. Raises where there is none — an absent read is not a gap of 0."""
        if not path.exists():
            message = f"{path} holds no leaderboard read; run `python -m kneemri.kaggle.board_read --live`"
            raise ValueError(message)
        held = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            leader=float(held["leader"]),
            standing=Standing(
                rank=int(held["rank"]),
                score=float(held["ours"]),
                field=int(held["field"]),
                tied=int(held["tied"]),
            ),
            taken=str(held["taken"]),
        )

    def stale(self, now: str, days: int = STALE_DAYS) -> list[str]:
        """Complaints about the AGE of this read, empty where it is fresh enough to plan against."""
        age = datetime.fromisoformat(now) - datetime.fromisoformat(self.taken)
        if age <= timedelta(days=days):
            return []
        return [
            f"the leaderboard read is {age.days} days old (taken {self.taken}), so its rank "
            f"{self.standing.rank} of {self.standing.field} is a fact about a board that has since seen "
            f"{5 * age.days} slots a competitor could spend"
        ]


def main() -> None:
    """Read the public board and record the gap: `python -m kneemri.kaggle.board_read --live`."""
    command = cli.Command(LeaderboardRead.__doc__, logger)
    command.parser.add_argument("--record", type=Path, default=RECORD, help="where the read is kept")
    command.parser.add_argument("--user", default=USERNAME, help="the Kaggle username to find us by")
    command.parser.add_argument(
        "--live",
        action="store_true",
        help="ask Kaggle for the whole public board (needs credentials, spends NO slot); without it the "
        "recorded read is printed and checked for age instead",
    )
    arguments = command.parsed()
    now = LeaderboardRead.now()

    if arguments.live:
        read = LeaderboardRead.taken_live(now, arguments.user).written(arguments.record)
        logger.info("recorded a fresh read into %s", read)

    held = LeaderboardRead.recorded(arguments.record)
    logger.info(
        "rank %d of %d, tied with %d at %.3f; leader %.3f, gap %.3f, taken %s",
        held.standing.rank,
        held.standing.field,
        held.standing.tied - 1,
        held.ours,
        held.leader,
        held.gap,
        held.taken,
    )
    for complaint in held.stale(now):
        logger.error("%s", complaint)
    if held.stale(now):
        raise SystemExit("the recorded gap is too old to plan against")


if __name__ == "__main__":
    main()
