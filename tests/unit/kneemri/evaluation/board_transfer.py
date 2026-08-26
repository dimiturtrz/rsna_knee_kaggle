import json
from pathlib import Path

import pytest

from kneemri.evaluation.board_transfer import REGIME_QUANTILES, BoardEntry, BoardPair, BoardTransfer

RECORD = [
    {"note": "a comment line carries no entry and must not be read as one"},
    {"id": "weak", "offline": 0.8500, "board": 0.860, "members": 1, "scope": "ensemble"},
    {"id": "strong", "offline": 0.8600, "board": 0.900, "members": 4, "scope": "ensemble"},
    # 0.0030 above `strong`: wide enough to divide by, under the floor the ruler can resolve.
    {"id": "close", "offline": 0.8630, "board": 0.890, "members": 5, "scope": "ensemble"},
    {"id": "averaged", "offline": 0.8200, "board": 0.888, "members": 5, "scope": "member_mean"},
    {"id": "silent", "offline": 0.8400, "board": 0.882, "members": 2, "scope": "unknown"},
    # An arm the holdout cannot ask, and an entry that blanked: one number each, no pair either way.
    {"id": "unasked", "offline": None, "board": 0.905, "members": 3, "scope": "ensemble"},
    {"id": "blanked", "offline": 0.8550, "board": None, "members": 2, "scope": "ensemble"},
]


def _record(tmp_path: Path) -> Path:
    path = tmp_path / "board.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in RECORD) + "\n", encoding="utf-8")
    return path


def test_entries(tmp_path: Path):
    """Only the ensemble scope is orderable, because a member mean is a different denominator."""
    entries = BoardTransfer(_record(tmp_path)).entries()

    assert [entry.id for entry in entries] == ["weak", "strong", "close"], "weakest holdout first"
    assert all(isinstance(entry, BoardEntry) for entry in entries)
    assert "averaged" not in {entry.id for entry in entries}, "a mean of members is not the ensemble"


def test_entries_reads_within_whichever_scope_it_is_asked_for(tmp_path: Path):
    """A scope other than the default is comparable to ITSELF — the four member_mean entries carry a
    reading the ensemble scope cannot: member quality up and board down, at held member count."""
    entries = BoardTransfer(_record(tmp_path), scope="member_mean").entries()

    assert [entry.id for entry in entries] == ["averaged"]


def test_dropped(tmp_path: Path):
    """A silent exclusion reads as 'everything was compared'; this names what was not."""
    assert BoardTransfer(_record(tmp_path)).dropped() == {"member_mean": 1, "unknown": 1}


def test_pairs(tmp_path: Path):
    """Every pair the holdout can order at all, and the ordering is the HOLDOUT's, not the board's."""
    pairs = BoardTransfer(_record(tmp_path)).pairs()

    assert {(pair.better.id, pair.worse.id) for pair in pairs} == {
        ("strong", "weak"),
        ("close", "weak"),
        ("close", "strong"),
    }
    assert all(pair.offline_delta > 0 for pair in pairs), "better by the holdout comes first"


def test_pairs_drops_a_gap_too_small_to_divide_by(tmp_path: Path):
    """A ratio over a vanishing denominator reports the denominator, so the pair is not formed."""
    path = tmp_path / "twins.jsonl"
    rows = [
        {"id": "one", "offline": 0.8600, "board": 0.900, "members": 2, "scope": "ensemble"},
        {"id": "two", "offline": 0.8601, "board": 0.910, "members": 2, "scope": "ensemble"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    assert BoardTransfer(path).pairs() == []


def test_inversions(tmp_path: Path):
    """The pair the board ordered the other way — the one that would have cost a submission slot."""
    inverted = BoardTransfer(_record(tmp_path)).inversions()

    assert [(pair.better.id, pair.worse.id) for pair in inverted] == [("close", "strong")]
    assert not inverted[0].resolved, "and the holdout never claimed that ordering: 0.0030 apart"


def test_compression(tmp_path: Path):
    """Board points per holdout point, over the pairs the ruler resolved at all."""
    # strong over weak: 0.040 board on 0.0100 holdout = 4.0. close over weak: 0.030 on 0.0130 = 2.31.
    # close over strong is UNRESOLVED at 0.0030 and contributes nothing, though it would read -3.3.
    assert BoardTransfer(_record(tmp_path)).compression() == pytest.approx(3.1538, abs=1e-4)


def test_floor_at(tmp_path: Path):
    """The cut points are the entries' own scores, so no level is invented to make a point."""
    transfer = BoardTransfer(_record(tmp_path))

    assert transfer.floor_at(0.0) == pytest.approx(0.8500), "lowest entry"
    assert transfer.floor_at(0.75) == pytest.approx(0.8630), "the strongest of three"


def test_compression_above(tmp_path: Path):
    """Restricting to the top entries drops the pair that spans the whole record."""
    transfer = BoardTransfer(_record(tmp_path))

    assert transfer.compression_above(0.0) == (2, pytest.approx(3.1538, abs=1e-4)), "both pairs"
    # Only `close` over `strong` starts at or above 0.8600, and it is UNRESOLVED at 0.0030.
    assert transfer.compression_above(0.8600) == (0, 0.0), "no resolved pair survives the floor"


def test_regimes(tmp_path: Path):
    """Every floor is reported with its own pair count, because the top cell is always the thinnest."""
    regimes = BoardTransfer(_record(tmp_path)).regimes()

    assert len(regimes) == len(REGIME_QUANTILES)
    assert [count for _, count, _ in regimes] == [0, 0], "nothing above the median is resolved here"
    assert all(floor >= 0.8500 for floor, _, _ in regimes), "floors come from the entries themselves"


def test_report(tmp_path: Path):
    """The report states the denominator it measured on, which is the number this project keeps losing."""
    report = BoardTransfer(_record(tmp_path)).report()

    assert "3 entries on the ensemble scope" in report
    assert "1 member_mean, 1 unknown" in report, "what was excluded is part of the result"
    assert "0/2 of the pairs the holdout resolved" in report
    assert "UNRESOLVED" in report, "an inversion under the floor is not a refutation"


def test_report_says_so_when_nothing_is_recorded(tmp_path: Path):
    """An empty record must not read as a clean sheet of zero inversions."""
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert "no entry carrying both numbers" in BoardTransfer(path).report()


def _pair(offline: float, board: float) -> BoardPair:
    return BoardPair(
        better=BoardEntry(id="better", offline=0.86, board=0.90, members=2),
        worse=BoardEntry(id="worse", offline=0.86 - offline, board=0.90 - board, members=2),
    )


def test_ratio():
    """One pair's own conversion: what the board paid for a point of holdout."""
    assert _pair(offline=0.01, board=0.02).ratio() == pytest.approx(2.0)


def test_offline_delta():
    """Positive by construction — the pair is built in the order the HOLDOUT ranked it."""
    assert _pair(offline=0.01, board=-0.02).offline_delta == pytest.approx(0.01)


def test_board_delta():
    """Signed, and the sign is the whole result: negative means the ruler chose the worse entry."""
    assert _pair(offline=0.01, board=-0.02).board_delta == pytest.approx(-0.02)


def test_inverted():
    """The board reversing the holdout is what costs a slot, so it is asked as its own question."""
    assert _pair(offline=0.01, board=-0.02).inverted
    assert not _pair(offline=0.01, board=0.02).inverted


def test_resolved():
    """A gap under the ruler's own noise floor is not a preference the board can contradict."""
    assert _pair(offline=0.01, board=0.02).resolved
    assert not _pair(offline=0.003, board=0.02).resolved


def test_entries_drops_a_row_carrying_only_one_of_the_two_numbers(tmp_path: Path):
    """A one-sided row is a legitimate record and never an entry.

    Reading it as one is not a wrong pair, it is a crash: `float(str(None))` raised ValueError and took
    the whole report down from the day the first blank was recorded until 2026-08-18.
    """
    transfer = BoardTransfer(_record(tmp_path))

    assert {entry.id for entry in transfer.entries()} == {"weak", "strong", "close"}
    assert transfer.report().startswith("3 entries")


def test_unmeasured(tmp_path: Path):
    """The dropped rows are NAMED, because a silent drop is how a blank submission stops being a result."""
    assert BoardTransfer(_record(tmp_path)).unmeasured() == ["unasked", "blanked"]
