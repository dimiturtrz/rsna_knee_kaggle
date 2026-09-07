"""The second ledger ruler: ten columns recut at 0.75, two held at the midpoint, both numbers kept."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from core.findings import TARGETS
from kneemri.evaluation.ledger75 import MIDPOINT_COLUMNS, LedgerSeventyFive
from kneemri.evaluation.scores import LabelScores
from kneemri.runs.holdout_record import HoldoutRecord

STUDIES = ["s1", "s2", "s3", "s4"]
GRADES = [0.94, 0.68, 0.68, 0.08]
"""Two of the reader's rungs on each side of the recut: 0.68 is positive at the midpoint and not at 0.75."""

RECUT_COLUMN = TARGETS[TARGETS.index("Effusion")]
HELD_COLUMN = MIDPOINT_COLUMNS[0]


def _truth() -> NDArray[np.float32]:
    """The same graded column repeated across all twelve findings."""
    return np.tile(np.array(GRADES, dtype=np.float32)[:, None], (1, len(TARGETS)))


def _member(directory: Path, name: str, ranked: str) -> Path:
    """A member that ranks `ranked` by the 0.94 row alone and every other finding backwards."""
    truth = _truth()
    predictions = np.tile(np.array([0.1, 0.9, 0.9, 0.9])[:, None], (1, len(TARGETS)))
    predictions[:, TARGETS.index(ranked)] = np.array([0.9, 0.2, 0.2, 0.1])
    path = directory / f"{name}.pt"
    HoldoutRecord(path).write(STUDIES, predictions, truth, epoch=5)
    return path


def test_reading(tmp_path: Path) -> None:
    """The kept arrays come back under their study names and the epoch they were written with."""
    reading = LedgerSeventyFive().reading(_member(tmp_path, "m1", RECUT_COLUMN))
    assert reading.studies == STUDIES
    assert reading.epoch == 5


def test_reading_refuses_a_member_with_no_record(tmp_path: Path) -> None:
    """A dark arm held nothing out, and the ruler must say so rather than price nothing."""
    with pytest.raises(FileNotFoundError, match="no holdout record"):
        LedgerSeventyFive().reading(tmp_path / "dark.pt")


def test_per_label(tmp_path: Path) -> None:
    """A member that ranks only the 0.94 row scores 1.0 where the cut is 0.75 and worse where it is not."""
    reading = LedgerSeventyFive().reading(_member(tmp_path, "m2", RECUT_COLUMN))
    per_label = LedgerSeventyFive().per_label(reading)
    assert per_label[RECUT_COLUMN] == pytest.approx(1.0)
    assert per_label[HELD_COLUMN] < 1.0


def test_macro(tmp_path: Path) -> None:
    """The macro is the NaN-skipping mean of exactly the columns `per_label` returns."""
    ruler = LedgerSeventyFive()
    reading = ruler.reading(_member(tmp_path, "m3", RECUT_COLUMN))
    assert ruler.macro(reading) == pytest.approx(float(np.nanmean(list(ruler.per_label(reading).values()))))


def test_row(tmp_path: Path) -> None:
    """A row is the pair: the midpoint macro the ledger holds and the recut macro beside it."""
    ruler = LedgerSeventyFive()
    path = _member(tmp_path, "m4", RECUT_COLUMN)
    row = ruler.row(path)
    reading = ruler.reading(path)
    midpoint = LabelScores().per_label(reading.predictions, reading.truth)
    assert row.name == "m4"
    assert row.studies == len(STUDIES)
    assert row.ledger == pytest.approx(float(np.nanmean(list(midpoint.values()))))
    assert row.ledger75 == pytest.approx(ruler.macro(reading))


def test_moved(tmp_path: Path) -> None:
    """The move is one member's own re-pricing, not a comparison against another member."""
    row = LedgerSeventyFive().row(_member(tmp_path, "m5", RECUT_COLUMN))
    assert row.moved() == pytest.approx(row.ledger75 - row.ledger)


def test_table(tmp_path: Path) -> None:
    """A dark arm beside the records must not take the sweep down, and the order is LEDGER75's."""
    strong = _member(tmp_path, "strong", RECUT_COLUMN)
    weak = _member(tmp_path, "weak", HELD_COLUMN)
    rows = LedgerSeventyFive().table([strong, weak, tmp_path / "dark.pt"])
    assert [row.name for row in rows] == ["strong", "weak"]
