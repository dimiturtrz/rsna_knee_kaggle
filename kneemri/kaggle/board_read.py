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

SERIES = Path("comp/leaderboard.jsonl")
"""Every read ever taken, appended — where `RECORD` holds only the latest and is OVERWRITTEN by it.

WHY A HISTORY AND NOT A SNAPSHOT. The plan that reaches 0.950 rests on two slopes: ours, fitted at
+0.0016/day over nine days of `comp/board.jsonl`, and the frontier's, for which this project held exactly
two points — 0.952 on 2026-08-25 and 0.952 on 2026-08-26 — because each read overwrote the last. Two
points is not a slope, and "the leader is flat" was the single unevidenced assumption underneath every
plan made on 2026-08-26. A read costs no submission slot, so the history is free and the assumption need
not stay one.
"""

TOP = 5
"""How many rows make the FRONTIER, as against the leader alone.

One team's best score moves in steps and sits still between them, so a leader series reads as flat for
days and then jumps — which is a fact about one team's submission schedule, not about the field. The mean
of the top five is the same measurement with four more draws, and it is the thing we are actually racing.
"""

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


class PublicBoard:
    """The competition's accuracy leaderboard as Kaggle ships it, and the ONE place this repo fetches it.

    `efficiency_read` reads the same board to join it against the efficiency track, and for a day the
    two modules each had their own download-and-unzip — two places to fix the next time the archive
    changes shape, and the BOM below is exactly that kind of surprise. `efficiency_read` now calls this
    and keeps only the efficiency-kernel pull, which is genuinely its own.
    """

    @staticmethod
    def text(into: Path) -> str:  # devtools-ignore: test-mirror
        """The whole board as CSV text. Credentials-only, so injected everywhere it is consumed.

        The DOWNLOAD, not the view: `competition_leaderboard_view` returns the board's first page and a
        team outside the top twenty is structurally absent from it. Neither call spends a slot.

        Decoded `utf-8-sig`: the CSV Kaggle ships carries a BOM, so under plain `utf-8` the first column
        is named `﻿Rank` and every reader raises `KeyError: 'Rank'` against a header that LOOKS right
        in a log.
        """
        import zipfile  # noqa: PLC0415

        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

        into.mkdir(parents=True, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        api.competition_leaderboard_download(COMPETITION, str(into))
        archive = max(into.glob(f"{COMPETITION}*.zip"), key=lambda path: path.stat().st_mtime)
        try:
            with zipfile.ZipFile(archive) as held:
                return held.read(held.namelist()[0]).decode("utf-8-sig")
        finally:
            archive.unlink()


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

    frontier: float = 0.0
    """The mean of the board's top `TOP` scores — the field's leading edge, not one team's best day.

    Defaulted rather than required because reads taken before 2026-08-26 have no frontier recorded, and a
    history that refuses to load its own older rows is a history that starts today.
    """

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
        best = sorted((row.score for row in rows), reverse=True)[:TOP]
        return cls(
            leader=max(row.score for row in rows),
            standing=Standing.of(username, rows),
            taken=now,
            frontier=sum(best) / len(best),
        )

    @staticmethod
    def _kaggle() -> list[BoardRow]:
        """The real round trip, kept in one place so every other path is credential-free."""
        import csv  # noqa: PLC0415
        import io  # noqa: PLC0415

        text = PublicBoard.text(RECORD.parent)
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
            **self.row(),
        }
        path.write_text(json.dumps(held, indent=2) + "\n", encoding="utf-8")
        return path

    def row(self) -> dict[str, float | int | str]:
        """This read as a flat record, shared by the snapshot and the history so they cannot drift apart."""
        return {
            "leader": self.leader,
            "frontier": round(self.frontier, 4),
            "ours": self.ours,
            "gap": round(self.gap, 4),
            "rank": self.standing.rank,
            "field": self.standing.field,
            "tied": self.standing.tied,
            "taken": self.taken,
        }

    def appended(self, path: Path = SERIES) -> Path:
        """Add this read to the history, unless a read with the same `taken` is already its last line.

        Idempotent on the instant rather than on the content, so re-running the command inside one second
        does not double-weight that second in a slope, while two genuinely different reads on the same day
        both survive — the fit is over instants and can carry as many as there are.
        """
        existing = self.series(path)
        if existing and existing[-1].taken == self.taken:
            return path
        with path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(self.row()) + "\n")
        return path

    @staticmethod
    def series(path: Path = SERIES) -> list["LeaderboardRead"]:
        """Every recorded read, oldest first. An absent history is an empty one, not an error."""
        if not path.exists():
            return []
        return [LeaderboardRead.of(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]

    @classmethod
    def of(cls, held: dict[str, float | int | str]) -> "LeaderboardRead":
        """One read from its flat record, tolerating the keys older rows do not carry."""
        return cls(
            leader=float(held["leader"]),
            standing=Standing(
                rank=int(held["rank"]),
                score=float(held["ours"]),
                field=int(held["field"]),
                tied=int(held["tied"]),
            ),
            taken=str(held["taken"]),
            frontier=float(held.get("frontier", 0.0)),
        )

    @classmethod
    def recorded(cls, path: Path = RECORD) -> "LeaderboardRead":
        """The last read, from the record. Raises where there is none — an absent read is not a gap of 0."""
        if not path.exists():
            message = f"{path} holds no leaderboard read; run `python -m kneemri.kaggle.board_read --live`"
            raise ValueError(message)
        return cls.of(json.loads(path.read_text(encoding="utf-8")))

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


MIN_READS = 3
"""Fewest reads this will fit a slope through.

Two points always fit a line exactly and report a slope with no residual, which is how "the leader is
flat" became a planning assumption off two identical numbers. Three is not enough either — it is the point
at which the fit can DISAGREE with its own points, and the report says how many it rests on so the reader
discounts it themselves.
"""

DAY = 86400.0


@dataclass(frozen=True)
class BoardTrend:
    """Two slopes fitted over the recorded history: ours, and the frontier's — and which is faster.

    THE CLOSING RATE IS THE ONLY NUMBER HERE THAT DECIDES ANYTHING. Our own climb is already measured off
    `comp/board.jsonl` and needs no board read; what a board read adds is whether the field above us is
    climbing at the same rate. A gap of 0.013 closes in eight days at +0.0016/day against a still frontier
    and never closes at all against one moving as fast as we are, and the difference between those two
    futures is not visible in any number this project held before the history existed.
    """

    reads: list[LeaderboardRead]

    def days(self) -> list[float]:
        """Days elapsed since the first read, one per read — the x-axis both slopes are fitted on."""
        if not self.reads:
            return []
        first = datetime.fromisoformat(self.reads[0].taken)
        return [(datetime.fromisoformat(read.taken) - first).total_seconds() / DAY for read in self.reads]

    def slope(self, values: list[float]) -> float:
        """Least-squares change per day, or NaN where the reads cannot support one.

        NaN rather than 0.0 for a short or degenerate history, because a slope of zero is a CLAIM — it is
        exactly the claim this whole module exists to stop being assumed — and an absent measurement must
        not be readable as a flat one.
        """
        spans = self.days()
        if len(spans) < MIN_READS or len(values) != len(spans):
            return float("nan")
        mean_day = sum(spans) / len(spans)
        spread = sum((day - mean_day) ** 2 for day in spans)
        if spread == 0.0:
            return float("nan")
        mean_value = sum(values) / len(values)
        return sum((day - mean_day) * (value - mean_value) for day, value in zip(spans, values, strict=True)) / spread

    def closing(self) -> float:
        """How fast the gap to the frontier shrinks per day: our slope minus the frontier's.

        Negative means the field above us is pulling away, and no amount of our own progress is closing
        anything. That is the reading a plan needs and the one a snapshot cannot give.
        """
        return self.slope([read.ours for read in self.reads]) - self.slope([read.frontier for read in self.reads])

    def report(self) -> list[str]:
        """The two slopes and the closing rate, with the count they rest on stated first."""
        spans = self.days()
        lines = [f"{len(self.reads)} recorded reads over {(spans[-1] if spans else 0.0):.1f} days"]
        if len(self.reads) < MIN_READS:
            lines.append(f"  fewer than {MIN_READS} reads — no slope is fitted, and none should be assumed")
            return lines
        ours, frontier = (
            self.slope([read.ours for read in self.reads]),
            self.slope([read.frontier for read in self.reads]),
        )
        closing = self.closing()
        lines.append(f"  ours {ours:+.5f}/day   frontier (top {TOP}) {frontier:+.5f}/day   closing {closing:+.5f}/day")
        latest = self.reads[-1]
        behind = latest.frontier - latest.ours
        if closing > 0:
            lines.append(f"  at that rate the {behind:+.4f} to the frontier closes in {behind / closing:.1f} days")
        else:
            lines.append(f"  the frontier is not being caught at this rate; {behind:+.4f} behind it and not closing")
        return lines


def main() -> None:
    """Read the public board and record the gap: `python -m kneemri.kaggle.board_read --live`."""
    command = cli.Command(LeaderboardRead.__doc__, logger)
    command.parser.add_argument("--record", type=Path, default=RECORD, help="where the read is kept")
    command.parser.add_argument("--user", default=USERNAME, help="the Kaggle username to find us by")
    command.parser.add_argument("--series", type=Path, default=SERIES, help="the appended read history")
    command.parser.add_argument(
        "--live",
        action="store_true",
        help="ask Kaggle for the whole public board (needs credentials, spends NO slot); without it the "
        "recorded read is printed and checked for age instead",
    )
    arguments = command.parsed()
    now = LeaderboardRead.now()

    if arguments.live:
        taken = LeaderboardRead.taken_live(now, arguments.user)
        logger.info("recorded a fresh read into %s", taken.written(arguments.record))
        # And into the HISTORY, which is what turns "the leader is flat" from an assumption into a fit.
        taken.appended(arguments.series)

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
    # The history, which answers the one question a snapshot structurally cannot: who is moving faster.
    command.report(BoardTrend(LeaderboardRead.series(arguments.series)).report())
    for complaint in held.stale(now):
        logger.error("%s", complaint)
    if held.stale(now):
        raise SystemExit("the recorded gap is too old to plan against")


if __name__ == "__main__":
    main()
