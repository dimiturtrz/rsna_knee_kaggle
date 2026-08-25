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

STALE_DAYS = 2
"""How old a read may be and still be worth planning against.

Derived, not chosen: the competition grants five submissions a day and the counter resets at 00:00 UTC,
so a read taken before yesterday sits behind at least ten spent slots — ours and every rival's. Two days
is the point past which the recorded gap describes a board that no longer exists.
"""


@dataclass(frozen=True)
class LeaderboardRead:
    """One look at the public board: what the leader holds, what we hold, and when it was taken."""

    leader: float
    ours: float
    taken: str
    """The UTC instant of the round trip, ISO-8601. Passed in rather than clocked inside, so the record
    a test writes is the record it asserts on."""

    rows: int
    """How many leaderboard rows the read actually covered — ONE PAGE, not the size of the field.

    The API paginates at twenty and hands back a next-page token. Page one is enough for the question
    this module asks, because Kaggle returns the board best-first so the leader is on it; it is NOT
    enough to say how many teams are competing, and the field is named for what was read rather than
    for what a reader might wish it meant.
    """

    @staticmethod
    def now() -> str:
        """The clock, in one place, so every caller stamps the same shape and no test reaches for its own."""
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @property
    def gap(self) -> float:
        """How far the leader is above us — the number the plan is actually made against."""
        return self.leader - self.ours

    @classmethod
    def taken_live(
        cls,
        ours: float,
        now: str,
        viewer: Callable[[], list[float]] | None = None,
    ) -> "LeaderboardRead":
        """Ask Kaggle for the public board, or ask whatever was injected in its place.

        Injected for the reason `board_check.LiveBoard` and `mount_check.LiveReading` give: a path
        reachable only through real credentials is a path with no test.
        """
        scores = viewer() if viewer is not None else cls._kaggle()
        if not scores:
            message = f"the leaderboard for {COMPETITION} came back empty, so no gap can be read from it"
            raise ValueError(message)
        return cls(leader=max(scores), ours=ours, taken=now, rows=len(scores))

    @staticmethod
    def _kaggle() -> list[float]:
        """The real round trip, kept in one place so every other path is credential-free."""
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

        api = KaggleApi()
        api.authenticate()
        viewed = api.competition_leaderboard_view(COMPETITION) or []
        return [float(entry.score) for entry in viewed if entry is not None and entry.score is not None]

    def written(self, path: Path = RECORD) -> Path:
        """Put the read where the next reader will find it without credentials."""
        held = {
            "why": "The public board as of `taken`, written by kneemri.kaggle.board_read. Never edit by "
            "hand and never quote from prose — re-read it, the gap decays every day slots are spent.",
            "leader": self.leader,
            "ours": self.ours,
            "gap": round(self.gap, 4),
            "taken": self.taken,
            "rows": self.rows,
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
            ours=float(held["ours"]),
            taken=str(held["taken"]),
            rows=int(held["rows"]),
        )

    def stale(self, now: str, days: int = STALE_DAYS) -> list[str]:
        """Complaints about the AGE of this read, empty where it is fresh enough to plan against."""
        age = datetime.fromisoformat(now) - datetime.fromisoformat(self.taken)
        if age <= timedelta(days=days):
            return []
        return [
            f"the leaderboard read is {age.days} days old (taken {self.taken}), so its {self.gap:.3f} gap "
            f"is a fact about a board that has since seen {5 * age.days} slots a competitor could spend"
        ]


def main() -> None:
    """Read the public board and record the gap: `python -m kneemri.kaggle.board_read --live`."""
    command = cli.Command(LeaderboardRead.__doc__, logger)
    command.parser.add_argument("--record", type=Path, default=RECORD, help="where the read is kept")
    command.parser.add_argument("--champion", type=Path, default=Path("comp/champion.json"), help="our score")
    command.parser.add_argument(
        "--live",
        action="store_true",
        help="ask Kaggle for the public board (needs credentials, spends NO slot); without it the "
        "recorded read is printed and checked for age instead",
    )
    arguments = command.parsed()
    now = LeaderboardRead.now()

    if arguments.live:
        ours = float(json.loads(arguments.champion.read_text(encoding="utf-8"))["board"])
        read = LeaderboardRead.taken_live(ours, now).written(arguments.record)
        logger.info("recorded a fresh read into %s", read)

    held = LeaderboardRead.recorded(arguments.record)
    logger.info(
        "leader %.3f, ours %.3f, gap %.3f against %d rows read, taken %s",
        held.leader,
        held.ours,
        held.gap,
        held.rows,
        held.taken,
    )
    for complaint in held.stale(now):
        logger.error("%s", complaint)
    if held.stale(now):
        raise SystemExit("the recorded gap is too old to plan against")


if __name__ == "__main__":
    main()
