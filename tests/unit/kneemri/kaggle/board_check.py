"""The submission record read against the board that made it — completeness, not correctness of recipe."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from kneemri.kaggle.board_check import BoardCensus, BoardRow, LiveBoard

CLEAN: Sequence[Mapping[str, object]] = [
    {"note": "a note carries commentary and no submission, so nothing here counts it"},
    {"id": "alpha_v1", "ref": "1", "date": "2026-08-20", "offline": 0.86, "board": 0.93, "scope": "ensemble"},
    {
        "id": "beta_v1",
        "ref": "2",
        "date": "2026-08-24",
        "offline": None,
        "board": 0.937,
        "scope": "ensemble",
        "members_named": ["calib_span_ff", "band_dense_s2026"],
    },
    {"ref": "3", "date": "2026-08-12", "board": 0.5, "scope": "excluded", "why": "kit failure"},
]
LIVE: dict[str, float | None] = {"1": 0.93, "2": 0.937, "3": 0.5}


def _census(
    tmp_path: Path,
    rows: Sequence[Mapping[str, object]] | None = None,
    prose: str = "alpha beta",
) -> BoardCensus:
    """A record and a ledger written into the test's own directory — it touches nothing it did not make."""
    record = tmp_path / "board.jsonl"
    record.write_text("".join(json.dumps(row) + "\n" for row in (rows if rows is not None else CLEAN)))
    ledger = tmp_path / "SUBMISSIONS.md"
    ledger.write_text(prose)
    return BoardCensus(record, ledger)


def test_parsed():
    """A JSONL line as a row, including the excluded line that carries no id and must still be reachable."""
    row = BoardRow.parsed(CLEAN[2])
    assert (row.identifier, row.ref, row.board, row.offline) == ("beta_v1", "2", 0.937, None)
    assert row.members_named == ("calib_span_ff", "band_dense_s2026")

    excluded = BoardRow.parsed(CLEAN[3])
    assert excluded.identifier == "3", "a row with no id is identified by its ref, never by an empty string"
    assert excluded.members_named == (), "no members named is an empty tuple, not a crash"


def test_disagrees():
    """A row read against a live score, including the null cases a float compare would crash on."""
    row = BoardRow.parsed(CLEAN[1])
    assert not row.disagrees(0.93)
    assert not row.disagrees(0.9302), "half a tick is transcription rounding, not a defect"
    assert row.disagrees(0.928)
    assert row.disagrees(None), "recorded a score, the board holds none"

    scoreless = BoardRow.parsed({"id": "none_v1", "ref": "9", "board": None})
    assert scoreless.disagrees(0.93), "recorded none, the board scored it"
    assert not scoreless.disagrees(None), "no score either side is agreement, not a defect"


def test_all_data():
    """The seats that held nothing out, by the roster's own naming rule."""
    assert BoardRow.parsed(CLEAN[2]).all_data() == ["calib_span_ff"], "the _ff suffix, and only it"
    named = {"id": "x", "ref": "9", "members_named": ["final_ws_e12"]}
    assert BoardRow.parsed(named).all_data() == ["final_ws_e12"], "the one member that predates the suffix"
    assert BoardRow.parsed(CLEAN[1]).all_data() == [], "an entry naming no members says nothing either way"


def test_rows(tmp_path: Path):
    """Every entry, the notes skipped — a note is commentary and would parse as a ref-less row."""
    assert [row.identifier for row in _census(tmp_path).rows()] == ["alpha_v1", "beta_v1", "3"]


def test_refless(tmp_path: Path):
    """A row no live listing can reach, which is what the whole join key exists to prevent."""
    assert _census(tmp_path).refless() == []
    assert _census(tmp_path, [{"id": "orphan_v1", "board": 0.9, "scope": "ensemble"}]).refless() == ["orphan_v1"]


def test_duplicated(tmp_path: Path):
    """One submission recorded twice is two points in every fit that reads this file."""
    assert _census(tmp_path).duplicated() == []
    twice = [*CLEAN, {"id": "alpha_v2", "ref": "1", "board": 0.93, "scope": "ensemble"}]
    assert _census(tmp_path, twice).duplicated() == ["1"]


def test_unpriceable(tmp_path: Path):
    """An offline number over a pool that has seen the holdout it was scored on."""
    assert _census(tmp_path).unpriceable() == [], "beta names an all-data seat and records no offline"
    priced = [{"id": "bad_v1", "ref": "1", "offline": 0.87, "board": 0.93, "members_named": ["calib_span_ff"]}]
    found = _census(tmp_path, priced).unpriceable()
    assert len(found) == 1 and "calib_span_ff" in found[0], "the offending seat is named, not just counted"


def test_unnamed(tmp_path: Path):
    """A priced entry the prose ledger never mentions; an excluded one is exempt by design."""
    assert _census(tmp_path).unnamed() == []
    assert _census(tmp_path, prose="alpha only").unnamed() == ["beta_v1"]
    assert "3" not in _census(tmp_path, prose="").unnamed(), "an excluded row earns no write-up"


def test_unrecorded(tmp_path: Path):
    """A submission that scored and reached no record — the failure this tool was written for."""
    assert _census(tmp_path).unrecorded(LIVE) == []
    assert _census(tmp_path).unrecorded({**LIVE, "99": 0.938}) == ["99"]


def test_phantom(tmp_path: Path):
    """A recorded ref the board does not hold — a typo in the join key reads as a whole record."""
    assert _census(tmp_path).phantom(LIVE) == []
    assert _census(tmp_path).phantom({"1": 0.93, "2": 0.937}) == ["3"]


def test_mismatched(tmp_path: Path):
    """A board number transcribed wrongly, and the null-against-scored case a float compare would miss."""
    assert _census(tmp_path).mismatched(LIVE) == []
    assert len(_census(tmp_path).mismatched({**LIVE, "1": 0.928})) == 1
    assert len(_census(tmp_path).mismatched({**LIVE, "1": None})) == 1, "recorded a score, the board holds none"


def test_refusals(tmp_path: Path):
    """Every defect at once, and the live half omitted entirely when no listing was taken."""
    assert _census(tmp_path).refusals(LIVE) == []
    assert _census(tmp_path).refusals(None) == []
    assert len(_census(tmp_path).refusals({**LIVE, "99": 0.938})) == 1
    assert _census(tmp_path).refusals(None) == [], "without a listing the offline rules still pass"


def test_taken():
    """The live listing through an injected lister — the real round trip is the only untested line."""
    board = LiveBoard.taken(lambda: [("1", "0.93"), ("4", "")])
    assert board.scores == {"1": 0.93, "4": None}, "a blank score is None, not 0.0"
