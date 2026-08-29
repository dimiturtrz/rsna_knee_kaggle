"""A null offline cell is filled only when every member of the entry held out what it is scored on."""

import json
from pathlib import Path

import numpy as np

from kneemri.kaggle.entry_offline import SCOPE_KEY, EntryOffline
from kneemri.runs.holdout_record import HoldoutRecord

LABELS = 12
STUDIES = 40
RNG = np.random.default_rng(2026)
CLEAN = "clean"
"""The one entry in the fixture board that every rule below lets through."""


def _member(path: Path, truth: np.ndarray) -> None:
    """One member's recorded predictions, and the checkpoint they were read from."""
    studies = [f"s{index}" for index in range(STUDIES)]
    predictions = truth * 2.0 + RNG.random((STUDIES, LABELS))
    HoldoutRecord(path).write(studies, predictions.astype(np.float64), truth.astype(np.float32), epoch=6)
    path.write_bytes(b"")


def _board(tmp_path: Path) -> Path:
    """One entry of each kind: fillable, full-fit, foreign, recordless, already priced, and a note."""
    rows = [
        {"note": "the record's own preamble, which is not an entry"},
        {"id": CLEAN, "ref": "1", "board": 0.93, "offline": None, "scope": "ensemble", "members_named": ["a", "b"]},
        {"id": "fitted", "ref": "2", "board": 0.94, "offline": None, "scope": "ensemble", "members_named": ["a_ff"]},
        {
            "id": "foreign",
            "ref": "3",
            "board": 0.92,
            "offline": None,
            "scope": "ensemble",
            "members_named": ["raptor_v4_swa"],
        },
        {"id": "gone", "ref": "4", "board": 0.91, "offline": None, "scope": "ensemble", "members_named": ["ghost"]},
        {"id": "priced", "ref": "5", "board": 0.9, "offline": 0.86, "scope": "ensemble", "members_named": ["a"]},
    ]
    path = tmp_path / "board.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _entries(tmp_path: Path) -> EntryOffline:
    """The fixture board over a processed root holding every member but `ghost`."""
    truth = (RNG.random((STUDIES, LABELS)) > 0.5).astype(np.float32)
    for name in ("a.pt", "b.pt", "a_ff.pt", "raptor_v4_swa.pt"):
        _member(tmp_path / name, truth)
    return EntryOffline(tmp_path, _board(tmp_path))


def test_rows(tmp_path: Path):
    """An entry already carrying a number, and the memberless note, are not this file's business."""
    rows = _entries(tmp_path).rows()

    assert sorted(row.identifier for row in rows) == ["clean", "fitted", "foreign", "gone"], "priced and note are out"


def test_declined(tmp_path: Path):
    """Each refusal names its own reason, and the clean entry gets none."""
    entries = _entries(tmp_path)
    reasons = {row.identifier: entries.declined(row) for row in entries.rows()}

    assert reasons[CLEAN] is None, "both members held out the studies they are scored on"
    assert str(reasons["fitted"]).startswith("in-sample"), "a _ff member has seen every holdout study"
    assert str(reasons["foreign"]).startswith("foreign-in-sample"), "measured overlap, not a guess about the name"
    assert str(reasons["gone"]).startswith("no-record"), "a member that wrote no predictions cannot be scored"


def test_scored(tmp_path: Path):
    """The number is a score on the shared studies, not a placeholder."""
    entries = _entries(tmp_path)
    clean = next(row for row in entries.rows() if row.identifier == CLEAN)

    assert 0.5 < entries.scored(clean) <= 1.0, "two members that track the truth beat chance"


def test_filled(tmp_path: Path):
    """Only the entry nothing declines gets a number."""
    filled = _entries(tmp_path).filled()

    assert list(filled) == [CLEAN], "three of the four nulls stay null"


def test_scopes(tmp_path: Path):
    """Every entry that stays null says why, so the null stops reading as an oversight."""
    scopes = _entries(tmp_path).scopes()

    assert sorted(scopes) == ["fitted", "foreign", "gone"], "each declined entry carries a reason"


def test_report(tmp_path: Path):
    """The survey closes with the counts it was asked for."""
    lines = _entries(tmp_path).report()

    assert lines[-1] == "1 fillable, 3 declined, of 4 entries with a null"


def test_written(tmp_path: Path):
    """The number and the scopes land in the record, and a row this file has no opinion about is untouched."""
    entries = _entries(tmp_path)

    changed = entries.written()

    rows = [json.loads(line) for line in entries.record.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["id"]: row for row in rows if "id" in row}
    assert changed == 4, "one filled, three scoped"
    assert by_id[CLEAN]["offline"] is not None, "the fillable entry now carries its number"
    assert SCOPE_KEY in by_id["foreign"], "the foreign entry now carries its reason"
    assert by_id["priced"]["offline"] == 0.86, "an entry that was already priced is left exactly as it was"
    assert "note" in rows[0], "the preamble survives a rewrite"
