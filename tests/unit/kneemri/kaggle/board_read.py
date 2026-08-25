import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kneemri.kaggle.board_read import BoardRow, LeaderboardRead, Standing

TAKEN = "2026-08-25T11:00:00+00:00"

US = "dimiturnt"


def _rows() -> list[BoardRow]:
    return [
        BoardRow(rank=1, score=0.953, members="brandon"),
        BoardRow(rank=2, score=0.938, members="mk,helper"),
        BoardRow(rank=3, score=0.938, members=f"{US},nobody"),
        BoardRow(rank=4, score=0.901, members="dimiturnt_other"),
    ]


def _read(taken: str = TAKEN) -> LeaderboardRead:
    return LeaderboardRead(
        leader=0.953,
        standing=Standing(rank=3, score=0.938, field=4, tied=2),
        taken=taken,
    )


def test_board_row_ours():
    """Our row is found by USERNAME, and a username that merely CONTAINS ours is not us.

    `TeamMemberUserNames` is comma-joined, so a substring test would read `dimiturnt_other` as us and
    report a rank belonging to a stranger.
    """
    assert _rows()[2].ours(US), "the row whose member list holds our username exactly"
    assert not _rows()[3].ours(US), "a longer username that merely starts with ours is not us"
    assert not _rows()[0].ours(US)


def test_of():
    """The standing, which is rank and the width of the band we are tied inside — not the gap."""
    standing = Standing.of(US, _rows())

    assert standing.rank == 3
    assert standing.score == 0.938
    assert standing.field == 4, "the WHOLE board, which is the number a page of twenty cannot give"
    assert standing.tied == 2, "us and everyone sharing our score, which is what makes a gap readable"


def test_of_refuses_a_board_without_us():
    """Absent from the board is a failed read, not last place — and never a rank of zero."""
    with pytest.raises(ValueError, match="no row on the public board lists"):
        Standing.of("someone-else", _rows())


def test_leaderboard_read_ours():
    """Our score comes off the BOARD, not out of our own champion record — two sources that can drift."""
    assert _read().ours == 0.938


def test_gap():
    """Still a subtraction and still real, but no longer the headline the plan is made against."""
    assert _read().gap == pytest.approx(0.015)


def test_taken_live():
    """The live round trip is INJECTED, so the whole path has a test without credentials.

    It takes the MAXIMUM rather than the first row: the API's ordering is theirs to change, and a gap
    read off a board that came back sorted the other way would be silently negative.
    """
    read = LeaderboardRead.taken_live(now=TAKEN, username=US, viewer=_rows)

    assert read.leader == 0.953, "the best score on the board, whatever order it arrived in"
    assert read.ours == 0.938, "read off the BOARD, not trusted from our own champion record"
    assert read.standing.rank == 3
    assert read.standing.field == 4, "the size of the FIELD, which the old one-page read could not see"
    assert read.taken == TAKEN, "the instant is passed in, so the record a test writes is what it asserts"


def test_taken_live_refuses_an_empty_board():
    """An empty listing is a failed read, and a failed read must not be written down as a gap of zero."""
    with pytest.raises(ValueError, match="came back empty"):
        LeaderboardRead.taken_live(now=TAKEN, username=US, viewer=list)


def test_written(tmp_path: Path):
    """The read is kept on disk so the next reader gets a rank AND a date rather than a number."""
    path = _read().written(tmp_path / "leaderboard.json")

    held = json.loads(path.read_text(encoding="utf-8"))

    assert held["gap"] == pytest.approx(0.015)
    assert held["rank"] == 3, "the headline, and the field it was read against travels beside it"
    assert held["field"] == 4
    assert held["tied"] == 2
    assert held["taken"] == TAKEN, "without this the file is prose again, which is the whole defect"


def test_recorded(tmp_path: Path):
    """A round trip through the file, and an ABSENT record refuses rather than reading as no gap."""
    path = _read().written(tmp_path / "leaderboard.json")

    held = LeaderboardRead.recorded(path)

    assert held.gap == pytest.approx(0.015)
    assert held.standing.rank == 3, "the rank survives the file, which is why the file gained the field"

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
