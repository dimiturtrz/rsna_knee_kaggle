"""Where do we stand on the OTHER leaderboard? `python -m kneemri.kaggle.efficiency_read --live`.

The Efficiency Prize is a second, separately-ranked track that every team is entered in automatically.
`board_read` reads the accuracy board; this reads the efficiency one and joins them, which is the only
way to see the quantity that matters: **at a fixed accuracy, how much rank is runtime worth?**

It exists because that question was answered by hand on 2026-08-11, written into `comp/README.md` and
`knee-5bv`, and was two weeks stale before anybody looked again — the field had grown from 905 ranked
teams to 2,238 and the efficiency leader had moved from 0.934 to 0.947. A hand analysis of a live board
decays silently, and every decision resting on it decays with it.

Costs NO submission slot: the accuracy board is a competition download, and the efficiency board is
published as the output of `ryanholbrook/rsna-knee-abnormalities-efficiency-lb`, which
`kernels_output` pulls.

THE TRACK IS ADDITIVE, AND THAT IS THE WHOLE REASON IT IS CHEAP TO ENTER. `additive` re-measures it
rather than trusting the note: teams carry an efficiency entry strictly BELOW their public best and
never above, so a deliberately cheap submission ADDS an efficiency candidate instead of displacing the
accurate one. If that ever stops being true, the cost of the track changes and this prints it.

THE EFFICIENCY BOARD IS A SNAPSHOT AND THE ACCURACY BOARD IS LIVE, WHICH FAKES THAT MECHANIC. A team
that scored better AFTER the snapshot reads as one holding a deliberately cheaper efficiency entry, and
the tell is one-sided: lag can only push the efficiency score DOWN, so "never above" is consistent with
pure staleness and proves nothing on its own. Measured 2026-08-25 against a snapshot two days old, the
naive rate was 34.8% (median 0.011 below) and the rate among teams whose last submission the snapshot
had actually seen was **20.5% (median 0.006)** — the track is additive, by half as much as the note
said. `settled` is the filter; `additive` reports through it and says how many rows it dropped.
"""

import io
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core import cli
from kneemri.kaggle import board_read

logger = logging.getLogger(__name__)

EFFICIENCY_KERNEL = "ryanholbrook/rsna-knee-abnormalities-efficiency-lb"
EFFICIENCY_FILE = "full_leaderboard.csv"

TEAM = "TeamName"
PUBLIC_SCORE = "Score"
EFFICIENCY_SCORE = "PublicScore"
"""The efficiency board names its score column `PublicScore` too, which is why the join renames nothing
and reads both columns by their own name — a `suffixes=` merge here produced `Score_x` and cost a read."""

EFFICIENCY_RANK = "EfficiencyRank"
SUBMITTED = "DateSubmitted"
LAST_SUBMITTED = "LastSubmissionDate"

STAMP = "%a %b %d %H:%M:%S %Y"
"""The efficiency board writes C-style stamps (`Sun Aug 23 11:52:24 2026`) while the accuracy board writes
ISO. Parsed explicitly rather than inferred, because a silent misparse here moves the snapshot date and
every rate computed against it."""

NEAR = 0.002
"""Half-width of the accuracy band `spread` calls "the same accuracy". Below the 0.0062 noise floor, so
two teams inside it are not distinguishable on the axis the accuracy board ranks by."""

FRONTIER = 20
"""How deep into the efficiency ranks `frontier` looks. A prize track pays a handful of places; the
question this answers is what accuracy is actually needed up there, not what the median team scores."""


@dataclass(frozen=True)
class EfficiencyBoard:
    """The two leaderboards side by side, joined on team name."""

    public: pd.DataFrame
    efficiency: pd.DataFrame

    def joined(self) -> pd.DataFrame:
        """The teams present on both boards, one row each, carrying both scores and both ranks.

        Joined on team name because the efficiency board publishes no team id — the one field the two
        share. Teams that renamed between the two snapshots drop out, and the count is printed so a
        collapsing join is visible rather than silently shrinking every rate computed from it.
        """
        return self.public.merge(self.efficiency, on=TEAM)

    def snapshot(self) -> pd.Timestamp:
        """When the efficiency board was last rebuilt, read as its newest submission.

        The board publishes no build stamp, so its own most recent row is the only evidence of how far
        behind the live accuracy board it is.
        """
        return pd.to_datetime(self.efficiency[SUBMITTED], format=STAMP).max()

    def settled(self) -> pd.DataFrame:
        """The joined teams whose last accuracy submission the efficiency snapshot had already seen.

        Only these rows can say anything about a team's CHOICE: for anyone who submitted afterwards, a
        lower efficiency score is the snapshot being old, not a second entry being held deliberately.
        """
        rows = self.joined()
        return rows[pd.to_datetime(rows[LAST_SUBMITTED]) <= self.snapshot()]

    def additive(self) -> str:
        """How many teams hold a cheaper efficiency entry than their accuracy entry — and how many hold a
        better one, which should be zero if the boards select independently.

        Measured over `settled` rows only. The naive rate over every joined row runs some 14 points
        higher and is mostly snapshot lag.
        """
        rows = self.settled()
        dropped = len(self.joined()) - len(rows)
        below = rows[EFFICIENCY_SCORE] < rows[PUBLIC_SCORE]
        above = int((rows[EFFICIENCY_SCORE] > rows[PUBLIC_SCORE]).sum())
        gap = float((rows[PUBLIC_SCORE] - rows[EFFICIENCY_SCORE])[below].median()) if below.any() else 0.0
        return (
            f"{int(below.sum())} of {len(rows)} settled teams ({100 * below.mean():.1f}%) carry an efficiency "
            f"entry BELOW their public best, median {gap:.3f} lower; {above} carry a better one "
            f"({dropped} teams dropped — they submitted after the snapshot, so lag would fake this)"
        )

    def spread(self, score: float, window: float = NEAR) -> str:
        """The efficiency ranks held by teams scoring within `window` of `score`.

        This is the measurement the track turns on. The accuracy band is narrow enough that the teams in
        it are equivalent on accuracy, so whatever rank range they span is bought with runtime alone.
        """
        rows = self.efficiency[(self.efficiency[EFFICIENCY_SCORE] - score).abs() <= window]
        if rows.empty:
            return f"no team scores within {window:.3f} of {score:.3f}"
        ranks = rows[EFFICIENCY_RANK]
        return (
            f"{len(rows)} teams within {window:.3f} of {score:.3f} hold efficiency ranks "
            f"{int(ranks.min())} to {int(ranks.max())} (median {int(ranks.median())}) — that span is runtime"
        )

    def frontier(self, top: int = FRONTIER) -> str:
        """The accuracy actually present in the top efficiency ranks, as a floor and a middle.

        The floor is the number to compare a cheap build against: a candidate scoring under it cannot
        place however fast it runs.
        """
        rows = self.efficiency[self.efficiency[EFFICIENCY_RANK] <= top]
        if rows.empty:
            return f"no team ranks in the top {top}"
        scores = rows[EFFICIENCY_SCORE]
        return (
            f"top {top} efficiency ranks score {scores.min():.3f} to {scores.max():.3f} "
            f"(median {scores.median():.3f}) — the floor is what a cheap build must clear"
        )

    def standing(self, team: str) -> list[str]:
        """Our own two rows, and how many teams beat us on efficiency while scoring WORSE than us.

        That last count is the honest size of the runtime debt: every one of those teams is behind us on
        accuracy and ahead of us anyway.
        """
        rows = self.joined()
        ours = rows[rows[TEAM] == team]
        if ours.empty:
            return [f"{team} is on neither board"]
        row = ours.iloc[0]
        cheaper = self.efficiency[
            (self.efficiency[EFFICIENCY_RANK] < row[EFFICIENCY_RANK])
            & (self.efficiency[EFFICIENCY_SCORE] < row[EFFICIENCY_SCORE])
        ]
        stale = pd.to_datetime(row[LAST_SUBMITTED]) > self.snapshot()
        lines = [
            f"{team}: accuracy rank {int(row['Rank'])} at {row[PUBLIC_SCORE]:.3f}, "
            f"efficiency rank {int(row[EFFICIENCY_RANK])} at {row[EFFICIENCY_SCORE]:.3f}",
            f"  {len(cheaper)} teams rank above us on efficiency while scoring BELOW us — runtime alone",
        ]
        if stale:
            lines.append(
                f"  OUR efficiency row is STALE: we submitted after the {self.snapshot():%Y-%m-%d %H:%M} "
                f"snapshot, so that rank describes a board we have since moved on"
            )
        return lines

    @staticmethod
    def taken(
        into: Path,
        download: Callable[[Path], tuple[str, Path]] | None = None,
    ) -> "EfficiencyBoard":
        """Both boards, downloaded into `into`. Injected for the reason `board_read` gives — a path that
        needs real credentials is a path with no test."""
        fetch = download if download is not None else EfficiencyBoard._kaggle
        public, efficiency = fetch(into)
        return EfficiencyBoard(pd.read_csv(io.StringIO(public)), pd.read_csv(efficiency))

    @staticmethod
    def _kaggle(into: Path) -> tuple[str, Path]:
        """The real round trips, kept in one place so every other path is credential-free.

        The accuracy board is fetched through `board_read.downloaded` rather than here: it is the same
        archive that module reads, and one home means the next surprise in its shape (a BOM, last time)
        is fixed once. What stays here is the efficiency kernel's output, which nothing else wants.
        """
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

        into.mkdir(parents=True, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        api.kernels_output(EFFICIENCY_KERNEL, str(into))
        return board_read.PublicBoard.text(into), into / EFFICIENCY_FILE


def main() -> None:
    """Read both leaderboards and price the efficiency track against our own standing on it."""
    command = cli.Command(EfficiencyBoard.__doc__, logger)
    command.parser.add_argument("--live", action="store_true", help="ask Kaggle (credentials, NO slot)")
    command.parser.add_argument("--team", type=str, default="Dimitar Terziev", help="the team to stand up")
    command.parser.add_argument("--into", type=Path, default=None, help="where the two boards land")
    arguments = command.parsed()

    if not arguments.live:
        raise SystemExit("nothing to read without --live; neither board has a local record")

    root = command.rooted(arguments)
    board = EfficiencyBoard.taken(arguments.into or root.processed() / "leaderboard")
    ours = board.joined()
    command.report([f"{len(board.public)} accuracy rows, {len(board.efficiency)} efficiency rows, {len(ours)} joined"])
    command.report([board.additive()])
    command.report(board.standing(arguments.team))
    mine = ours[ours[TEAM] == arguments.team]
    if not mine.empty:
        command.report([board.spread(float(mine[EFFICIENCY_SCORE].iloc[0]))])
    command.report([board.frontier()])


if __name__ == "__main__":
    main()
