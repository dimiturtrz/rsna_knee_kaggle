"""Both leaderboards joined, on tables built here rather than downloaded."""

from pathlib import Path

import pandas as pd

from kneemri.kaggle.efficiency_read import (
    EFFICIENCY_RANK,
    EFFICIENCY_SCORE,
    LAST_SUBMITTED,
    PUBLIC_SCORE,
    SUBMITTED,
    TEAM,
    EfficiencyBoard,
)

OURS = "ours"
SNAPSHOT = "Sun Aug 23 22:30:36 2026"
BEFORE = "2026-08-22 10:00:00"
AFTER = "2026-08-25 03:36:22"


def _boards(rows: list[tuple[str, float, float, int]], last: str = BEFORE) -> EfficiencyBoard:
    """One row per team: its name, its accuracy score, its efficiency score and its efficiency rank.

    Every team's last accuracy submission defaults to one the snapshot had seen, so a test that is not
    about lag is not silently measuring it.
    """
    public = pd.DataFrame(
        [
            {TEAM: name, PUBLIC_SCORE: score, "Rank": index + 1, LAST_SUBMITTED: last}
            for index, (name, score, _, _) in enumerate(rows)
        ]
    )
    efficiency = pd.DataFrame(
        [
            {TEAM: name, EFFICIENCY_SCORE: score, EFFICIENCY_RANK: rank, SUBMITTED: SNAPSHOT}
            for name, _, score, rank in rows
        ]
    )
    return EfficiencyBoard(public, efficiency)


def test_joined():
    """Only the teams on BOTH boards survive, so a name present on one is not counted as a team."""
    board = _boards([(OURS, 0.938, 0.933, 95), ("rival", 0.947, 0.947, 1)])
    board.efficiency.loc[len(board.efficiency)] = {TEAM: "ghost", EFFICIENCY_SCORE: 0.9, EFFICIENCY_RANK: 500}

    joined = board.joined()

    assert set(joined[TEAM]) == {OURS, "rival"}, "the efficiency-only team drops out"
    assert len(joined) == 2


def test_additive():
    """The mechanic the track rests on: entries below a team's public best, and none above it."""
    board = _boards([(OURS, 0.938, 0.933, 95), ("rival", 0.947, 0.947, 1), ("cheap", 0.940, 0.940, 3)])

    line = board.additive()

    assert "1 of 3 settled teams (33.3%)" in line, line
    assert "median 0.005 lower" in line, line
    assert "0 carry a better one" in line, line
    assert "0 teams dropped" in line, "every team here submitted before the snapshot"


def test_snapshot():
    """The board's own newest submission is the only evidence of when it was built."""
    board = _boards([(OURS, 0.938, 0.933, 95), ("rival", 0.947, 0.947, 1)])
    board.efficiency.loc[0, SUBMITTED] = "Fri Aug 14 09:00:00 2026"

    assert board.snapshot() == pd.Timestamp("2026-08-23 22:30:36"), "the NEWEST row, not the first one"


def test_settled():
    """A team that submitted after the snapshot is dropped — its lower efficiency score is only lag."""
    board = _boards([(OURS, 0.938, 0.933, 95), ("rival", 0.947, 0.947, 1)])
    board.public.loc[0, LAST_SUBMITTED] = AFTER

    settled = board.settled()

    assert list(settled[TEAM]) == ["rival"], "only the team the snapshot had seen survives"
    assert "1 teams dropped" in board.additive()
    assert "0 of 1 settled teams" in board.additive(), "and the surviving team holds no cheaper entry"


def test_spread():
    """At one accuracy the rank range is runtime, and the band is what decides who is at that accuracy."""
    board = _boards([(OURS, 0.938, 0.933, 95), ("fast", 0.933, 0.934, 6), ("slow", 0.933, 0.932, 1667)])

    line = board.spread(0.933)

    assert "3 teams" in line and "ranks 6 to 1667" in line, line
    assert "1 teams" in board.spread(0.933, window=0.0), "a zero band admits only the exact score"
    assert board.spread(0.5).startswith("no team"), "an accuracy nobody holds is said so, not divided by zero"


def test_frontier():
    """The accuracy floor in the paying ranks — a cheap build under it cannot place however fast it is."""
    board = _boards([(OURS, 0.938, 0.933, 95), ("a", 0.947, 0.947, 1), ("b", 0.920, 0.918, 4)])

    line = board.frontier(top=20)

    assert "0.918 to 0.947" in line, line
    assert board.frontier(top=0) == "no team ranks in the top 0"


def test_standing():
    """Our two ranks, plus the count of teams that beat us on efficiency while scoring worse."""
    board = _boards([(OURS, 0.938, 0.933, 95), ("worse but faster", 0.930, 0.925, 40), ("better", 0.947, 0.947, 1)])

    lines = board.standing(OURS)

    assert "accuracy rank 1 at 0.938" in lines[0] and "efficiency rank 95 at 0.933" in lines[0], lines
    assert "1 teams rank above us" in lines[1], "only the worse-but-faster team counts"
    assert len(lines) == 2, "nobody here submitted after the snapshot, so no staleness line"
    assert board.standing("absent") == ["absent is on neither board"]

    board.public.loc[0, LAST_SUBMITTED] = AFTER

    assert "OUR efficiency row is STALE" in board.standing(OURS)[2], "a rank we have moved on from says so"


def test_taken(tmp_path: Path):
    """The download is injected, so the whole read is exercised without credentials."""
    built = _boards([(OURS, 0.938, 0.933, 95)])

    def _written(into: Path) -> tuple[Path, Path]:
        into.mkdir(parents=True, exist_ok=True)
        public, efficiency = into / "public.csv", into / "efficiency.csv"
        built.public.to_csv(public, index=False)
        built.efficiency.to_csv(efficiency, index=False)
        return public, efficiency

    board = EfficiencyBoard.taken(tmp_path / "boards", download=_written)

    assert list(board.public[TEAM]) == [OURS], "a plain csv is read as itself, not unzipped"
    assert int(board.efficiency[EFFICIENCY_RANK].iloc[0]) == 95
