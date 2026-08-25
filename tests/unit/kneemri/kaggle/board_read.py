import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kneemri.kaggle.board_read import LeaderboardRead

TAKEN = "2026-08-25T11:00:00+00:00"


def _read(taken: str = TAKEN) -> LeaderboardRead:
    return LeaderboardRead(leader=0.953, ours=0.938, taken=taken, rows=20)


def test_gap():
    """The one number nothing in the repo could read, and it is a subtraction, not a stored field."""
    assert _read().gap == pytest.approx(0.015)


def test_taken_live():
    """The live round trip is INJECTED, so the whole path has a test without credentials.

    It takes the MAXIMUM rather than the first row: the API's ordering is theirs to change, and a gap
    read off a board that came back sorted the other way would be silently negative.
    """
    read = LeaderboardRead.taken_live(ours=0.938, now=TAKEN, viewer=lambda: [0.901, 0.953, 0.944])

    assert read.leader == 0.953, "the best score on the board, whatever order it arrived in"
    assert read.rows == 3, "what was READ, which is one page and not the size of the field"
    assert read.taken == TAKEN, "the instant is passed in, so the record a test writes is what it asserts"


def test_taken_live_refuses_an_empty_board():
    """An empty listing is a failed read, and a failed read must not be written down as a gap of zero."""
    with pytest.raises(ValueError, match="came back empty"):
        LeaderboardRead.taken_live(ours=0.938, now=TAKEN, viewer=list)


def test_written(tmp_path: Path):
    """The read is kept on disk so the next reader gets a number AND a date rather than a number."""
    path = _read().written(tmp_path / "leaderboard.json")

    held = json.loads(path.read_text(encoding="utf-8"))

    assert held["gap"] == pytest.approx(0.015)
    assert held["taken"] == TAKEN, "without this the file is prose again, which is the whole defect"


def test_recorded(tmp_path: Path):
    """A round trip through the file, and an ABSENT record refuses rather than reading as no gap."""
    path = _read().written(tmp_path / "leaderboard.json")

    assert LeaderboardRead.recorded(path).gap == pytest.approx(0.015)

    with pytest.raises(ValueError, match="holds no leaderboard read"):
        LeaderboardRead.recorded(tmp_path / "absent.json")


def test_stale():
    """The decay this module exists for: the pinned gap was 0.008, was 0.015, and nothing was looking."""
    fresh = _read().stale(now="2026-08-26T11:00:00+00:00")

    assert fresh == [], "one day old is a board that has seen five slots, which is still plannable"

    old = _read().stale(now="2026-08-29T11:00:00+00:00")

    assert len(old) == 1
    assert "4 days old" in old[0]
    assert "20 slots" in old[0], "the complaint prices the staleness in the units that cause it"


def test_now():
    """One clock, so no caller reaches for its own and no two records carry different shapes."""
    stamped = datetime.fromisoformat(LeaderboardRead.now())

    assert stamped.tzinfo is not None, "a naive stamp cannot be differenced against another machine's"
    assert stamped.tzinfo.utcoffset(stamped) == timezone.utc.utcoffset(stamped)
