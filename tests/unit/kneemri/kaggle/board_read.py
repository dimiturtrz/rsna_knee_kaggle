import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kneemri.kaggle import board_read
from kneemri.kaggle.board_read import BoardRow, BoardTrend, LeaderboardRead, PublicBoard, Standing

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


def test_standing_of():
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


def _trend(oursides: list[float], frontiers: list[float]) -> BoardTrend:
    return BoardTrend(
        [
            LeaderboardRead(
                leader=front,
                standing=Standing(rank=3, score=ours, field=4, tied=2),
                taken=f"2026-08-2{day}T11:00:00+00:00",
                frontier=front,
            )
            for day, (ours, front) in enumerate(zip(oursides, frontiers, strict=True), start=1)
        ]
    )


def test_row():
    """The flat record the snapshot and the history SHARE, so the two files cannot drift apart."""
    row = _read().row()

    assert row["rank"] == 3
    assert row["taken"] == TAKEN
    assert row["frontier"] == 0.0, "an unrecorded frontier is zero, never silently the leader"


def test_leaderboard_read_of():
    """A read rebuilt from a flat record, tolerating the `frontier` older rows do not carry."""
    rebuilt = LeaderboardRead.of(_read().row())

    assert rebuilt.ours == 0.938
    assert rebuilt.taken == TAKEN

    older = _read().row()
    del older["frontier"]

    assert LeaderboardRead.of(older).frontier == 0.0, "a history must load rows written before the column"


def test_appended(tmp_path: Path):
    """The history GROWS where the snapshot is overwritten — and one instant lands at most once."""
    path = tmp_path / "leaderboard.jsonl"
    _read().appended(path)
    _read().appended(path)

    assert len(LeaderboardRead.series(path)) == 1, "one instant re-read must not double-weight it in a slope"

    _read(taken="2026-08-26T11:00:00+00:00").appended(path)

    assert len(LeaderboardRead.series(path)) == 2, "a genuinely later read is a second point"


def test_series(tmp_path: Path):
    """An absent history is an EMPTY one — the first read must not have to create it by hand."""
    assert LeaderboardRead.series(tmp_path / "absent.jsonl") == []


def test_days():
    """The x-axis both slopes are fitted on, in days since the first read rather than in row numbers."""
    assert _trend([0.93, 0.94, 0.95], [0.95, 0.95, 0.95]).days() == [0.0, 1.0, 2.0]


def test_slope():
    """A fitted change per day — and NaN, never zero, where the history cannot support one.

    Zero is a CLAIM, and it is exactly the claim ("the leader is flat") this history exists to stop being
    assumed off two identical numbers.
    """
    trend = _trend([0.930, 0.932, 0.934], [0.950, 0.950, 0.950])

    assert trend.slope([read.ours for read in trend.reads]) == pytest.approx(0.002)
    assert trend.slope([read.frontier for read in trend.reads]) == pytest.approx(0.0, abs=1e-12)

    short = _trend([0.93, 0.94], [0.95, 0.95])

    assert short.slope([read.ours for read in short.reads]) != short.slope([read.ours for read in short.reads]), (
        "two points fit a line exactly and must report NaN rather than a slope"
    )


def test_days_of_an_empty_history_is_empty():
    assert BoardTrend([]).days() == []


def test_slope_of_reads_all_taken_at_one_instant_is_nan():
    """Three reads at one instant span no time, so they measure no rate — NaN, not a division by zero."""
    one = _read()
    trend = BoardTrend([one, one, one])
    assert math.isnan(trend.slope([0.93, 0.94, 0.95]))


BOARD = "Rank,TeamName,Score,TeamMemberUserNames\n1,lead,0.953,brandon\n3,us,0.938,dimiturnt\n"


class _Api:
    """Writes the zip Kaggle's download writes, so the read path runs exactly as it does live."""

    def competition_leaderboard_download(self, slug: str, into: str) -> None:
        with zipfile.ZipFile(Path(into) / f"{slug}.zip", "w") as archive:
            archive.writestr("board.csv", BOARD.encode("utf-8-sig"))


def test_text_strips_the_bom_and_leaves_no_archive_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(board_read.Kaggle, "authenticated", staticmethod(_Api))
    into = tmp_path / "comp"
    text = PublicBoard.text(into)
    assert text.startswith("Rank,"), "a BOM would rename the first column"
    assert list(into.iterdir()) == []


def test_kaggle_parses_every_line_of_the_download(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(PublicBoard, "text", staticmethod(lambda into: BOARD))
    rows = LeaderboardRead._kaggle()
    assert rows == [
        BoardRow(rank=1, score=0.953, members="brandon"),
        BoardRow(rank=3, score=0.938, members="dimiturnt"),
    ]


def test_closing():
    """The only number that decides anything: our climb MINUS the frontier's."""
    assert _trend([0.930, 0.932, 0.934], [0.950, 0.950, 0.950]).closing() == pytest.approx(0.002)
    assert _trend([0.930, 0.932, 0.934], [0.950, 0.953, 0.956]).closing() == pytest.approx(-0.001)


def test_report():
    """The count comes FIRST, and a history too short to fit says so instead of printing a flat slope."""
    lines = _trend([0.930, 0.932, 0.934], [0.950, 0.950, 0.950]).report()

    assert "3 recorded reads" in lines[0]
    assert "closing +0.00200/day" in lines[1]
    assert "closes in" in lines[2]

    losing = _trend([0.930, 0.932, 0.934], [0.950, 0.953, 0.956]).report()

    assert "not being caught" in losing[2], "a frontier pulling away must never read as a countdown"

    assert "no slope is fitted" in _trend([0.93, 0.94], [0.95, 0.95]).report()[1]
