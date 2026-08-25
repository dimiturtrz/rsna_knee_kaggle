"""An entry is offered as a pricing base only if EVERY member of it can still be scored."""

import json
from pathlib import Path

import numpy as np

from kneemri.kaggle.priceable_entries import PriceableEntries
from kneemri.runs.holdout_record import HoldoutRecord

LABELS = 12
STUDIES = 40
RNG = np.random.default_rng(2026)


def _member(path: Path, truth: np.ndarray) -> None:
    """One member's record, and the checkpoint it was read from — a record alone is a dead run."""
    studies = [f"s{index}" for index in range(STUDIES)]
    predictions = truth * 2.0 + RNG.random((STUDIES, LABELS))
    HoldoutRecord(path).write(studies, predictions.astype(np.float64), truth.astype(np.float32), epoch=6)
    path.write_bytes(b"")


def _board(tmp_path: Path) -> Path:
    """A board holding one fully scoreable entry, one that is not, and two rows that are not entries.

    The last two are the reason `entries` cannot simply map `from_board` over the file: the real record
    opens with prose notes carrying no `id`, and 31 of its rows record no members at all.
    """
    rows = [
        {"note": "the record's own preamble, which is not an entry"},
        {"id": "whole", "kernel": "k", "board": 0.933, "members_named": ["shipped", "spare"], "scope": "ensemble"},
        {"id": "dark", "kernel": "k", "board": 0.938, "members_named": ["shipped", "ghost"], "scope": "ensemble"},
        {"id": "ruined", "kernel": "k", "board": 0.94, "scope": "ensemble"},
    ]
    path = tmp_path / "board.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _priceable(tmp_path: Path) -> PriceableEntries:
    """Both members of the fully scoreable entry on disk; the other entry's `ghost` deliberately absent."""
    truth = (RNG.random((STUDIES, LABELS)) > 0.5).astype(np.float32)
    for name in ("shipped.pt", "spare.pt"):
        _member(tmp_path / name, truth)
    return PriceableEntries(tmp_path, _board(tmp_path))


def test_entries(tmp_path: Path):
    """A row nothing can be rebuilt from is skipped, not raised on — a survey has no business failing."""
    entries = _priceable(tmp_path).entries()

    assert sorted(entry.entry for entry in entries) == ["dark", "whole"], "the note and the memberless row are gone"


def test_fully(tmp_path: Path):
    """`dark` scored higher and prices nothing, which is the whole shape of the real board."""
    whole = _priceable(tmp_path).fully(minimum=2)

    assert [entry.entry for entry in whole] == ["whole"], "an entry is suggested only if EVERY member scores"


def test_fully_refuses_an_entry_below_the_caller_s_own_minimum(tmp_path: Path):
    """The threshold is passed in, so a suggestion can never name an entry the caller would then refuse."""
    whole = _priceable(tmp_path).fully(minimum=3)

    assert whole == [], "a two-member roster is no answer to a caller that needs three"


def test_suggestion(tmp_path: Path):
    """The line names an entry and its score, so the reader does not go and survey the board by hand."""
    line = _priceable(tmp_path).suggestion(minimum=2)

    assert "--against whole" in line
    assert "0.933" in line, "the score is shown, since the caller is trading it against the champion's"
    assert "dark" not in line, "the higher entry is not offered, because it cannot be priced against"


def test_suggestion_says_so_when_nothing_can_price(tmp_path: Path):
    """The empty case is a statement, not a blank — an absent suggestion reads as an absent problem."""
    line = PriceableEntries(tmp_path, _board(tmp_path)).suggestion(minimum=2)

    assert "NO board entry is fully scoreable" in line
