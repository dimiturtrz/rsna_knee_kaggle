"""The uncertain-mention surface: which cells it drops, what it refuses to join, and the re-ranking."""

from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.unk_surface import MINIMUM_STUDIES, UNCERTAIN, UnkReading, UnkSurface

UNCERTAIN_EVERY = 4
"""One cell in four, so a fixture record's uncertain share clears `MINIMUM_UNCERTAIN` as v2 does."""

STUDIES = 400
"""Above `MINIMUM_STUDIES`, so a fixture record is read rather than refused for being a smoke run."""

CONFIDENT_NEGATIVE = 0.08
CONFIDENT_POSITIVE = 0.94


def _truth(uncertain_every: int = 0) -> np.ndarray:
    """Alternating confident labels, with every `uncertain_every`-th study set to the UNK constant."""
    column = np.where(np.arange(STUDIES) % 2 == 0, CONFIDENT_POSITIVE, CONFIDENT_NEGATIVE)
    if uncertain_every:
        column[::uncertain_every] = UNCERTAIN
    return np.repeat(column[:, None], len(TARGETS), axis=1)


def _predictions(truth: np.ndarray) -> np.ndarray:
    """A member that reads the confident cells perfectly and the uncertain ones as positive."""
    return np.where(np.isclose(truth, UNCERTAIN), 1.0, truth)


def _record(tmp_path: Path, member: str, *, inverted: bool = False) -> Path:
    """A member record on the v2 ruler; `inverted` reads every cell backwards, so it ranks last."""
    truth = _truth(UNCERTAIN_EVERY)
    predictions = _predictions(truth)
    path = tmp_path / f"{member}.holdout.npz"
    np.savez(path, predictions=1.0 - predictions if inverted else predictions, truth=truth)
    return path


def test_move() -> None:
    reading = UnkReading(member="m", area=0.80, excluded=0.85, uncertain_share=0.27, columns=12)
    assert reading.move() == pytest.approx(0.05)


def test_line() -> None:
    line = UnkReading(member="m", area=0.80, excluded=0.85, uncertain_share=0.27, columns=12).line()
    assert "0.8000" in line
    assert "0.8500" in line
    assert "27.0%" in line


def test_is_uncertain() -> None:
    flagged = UnkSurface().is_uncertain(np.array([UNCERTAIN, 0.08, 0.68, 0.94]))
    assert flagged.tolist() == [True, False, False, False]


def test_reading() -> None:
    truth = _truth(uncertain_every=4)
    reading = UnkSurface().reading("m", _predictions(truth), truth)
    assert reading is not None
    assert reading.columns == len(TARGETS)
    assert reading.excluded == 1.0
    assert reading.area < reading.excluded
    assert reading.uncertain_share == 0.25


def test_reading_single_class() -> None:
    """No column resolves when every truth cell is the same label, so there is nothing to report."""
    truth = np.full((STUDIES, len(TARGETS)), CONFIDENT_POSITIVE)
    assert UnkSurface().reading("m", truth.copy(), truth) is None


def test_of(tmp_path: Path) -> None:
    reading = UnkSurface().of(_record(tmp_path, "member_a"))
    assert reading is not None
    assert reading.member == "member_a"


def test_of_refuses_short(tmp_path: Path) -> None:
    """A frame below the smoke-run floor is not a member reading and is refused rather than scored."""
    truth = _truth(UNCERTAIN_EVERY)[: MINIMUM_STUDIES - 1]
    path = tmp_path / "smoke.holdout.npz"
    np.savez(path, predictions=_predictions(truth), truth=truth)
    assert UnkSurface().of(path) is None


def test_of_refuses_mismatched_shape(tmp_path: Path) -> None:
    truth = _truth(UNCERTAIN_EVERY)
    path = tmp_path / "ragged.holdout.npz"
    np.savez(path, predictions=_predictions(truth)[:, :3], truth=truth)
    assert UnkSurface().of(path) is None


def test_of_refuses_moved_ruler(tmp_path: Path) -> None:
    """A truth block with no uncertain cells was scored by a re-mapped table, not by v2.

    Its area is on a different ruler, so ranking it beside the ledger's members is the defect that
    printed `recal_holdannot_s2026`'s forbidden number.
    """
    truth = _truth()
    path = tmp_path / "remapped.holdout.npz"
    np.savez(path, predictions=_predictions(truth), truth=truth)
    assert UnkSurface().of(path) is None


def test_sweep(tmp_path: Path) -> None:
    _record(tmp_path, "reads")
    _record(tmp_path, "backwards", inverted=True)
    readings = UnkSurface().sweep(tmp_path)
    assert [row.member for row in readings] == ["reads", "backwards"]


def test_inversions() -> None:
    readings = [
        UnkReading(member="a", area=0.90, excluded=0.80, uncertain_share=0.2, columns=12),
        UnkReading(member="b", area=0.85, excluded=0.88, uncertain_share=0.2, columns=12),
        UnkReading(member="c", area=0.80, excluded=0.70, uncertain_share=0.2, columns=12),
    ]
    assert UnkSurface.inversions(readings) == (1, 3)
    assert UnkSurface.inversions(readings[:1]) == (0, 0)


def test_report(tmp_path: Path) -> None:
    lines = UnkSurface().report(UnkSurface().sweep(tmp_path))
    assert lines == ["no holdout record carried both a predictions and a truth block"]
    _record(tmp_path, "reads")
    _record(tmp_path, "backwards", inverted=True)
    body = "\n".join(UnkSurface().report(UnkSurface().sweep(tmp_path)))
    assert "2 members" in body
    assert "re-ranking" in body
    assert "NOT a gain" in body
