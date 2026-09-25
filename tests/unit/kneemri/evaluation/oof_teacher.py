"""The teacher table is a concatenation, so the tests state what may and may not be concatenated."""

from pathlib import Path

import numpy as np
import pytest

from kneemri.evaluation.oof_teacher import BlockStats, OutOfFoldTeacher
from kneemri.teacher_file import TeacherFile

TARGETS = 12
"""Columns a teacher file carries, one per label."""


def _fold(directory: Path, name: str, studies: list[str]) -> Path:
    path = directory / f"{name}.holdout.npz"
    scores = np.full((len(studies), TARGETS), float(len(studies)), dtype=np.float32)
    np.savez(path, studies=np.asarray(studies, dtype="<U64"), predictions=scores, truth=np.zeros_like(scores))
    return path


def test_readings(tmp_path: Path) -> None:
    parts = (_fold(tmp_path, "oof_fold1_s2026", ["a", "b"]), _fold(tmp_path, "oof_fold2_s2026", ["c"]))
    names, scores = OutOfFoldTeacher(parts).readings()
    assert names == ["a", "b", "c"]
    assert scores.shape == (3, TARGETS)
    assert scores[0, 0] == pytest.approx(2.0)
    assert scores[2, 0] == pytest.approx(1.0)


def test_readings_refuses_overlapping_folds(tmp_path: Path) -> None:
    parts = (_fold(tmp_path, "oof_fold1_s2026", ["a", "b"]), _fold(tmp_path, "oof_fold2_s2026", ["b"]))
    with pytest.raises(ValueError, match="not disjoint"):
        OutOfFoldTeacher(parts).readings()


def test_write(tmp_path: Path) -> None:
    parts = (_fold(tmp_path, "oof_fold1_s2026", ["a", "b"]), _fold(tmp_path, "oof_fold2_s2026", ["c"]))
    out = tmp_path / "teacher.npz"
    lines = OutOfFoldTeacher(parts).write(out)
    stored = np.load(out, allow_pickle=True)
    assert [str(name) for name in stored[TeacherFile.IDENTIFIERS]] == ["a", "b", "c"]
    assert stored[TeacherFile.SCORES].shape == (3, TARGETS)
    assert any("3 training studies" in line for line in lines)


def _block(directory: Path, name: str, score: float) -> Path:
    """A block whose every cell reads the same, so a stat over it is the value asked for."""
    path = directory / f"{name}.holdout.npz"
    np.savez(
        path,
        studies=np.asarray([name], dtype="<U64"),
        predictions=np.full((1, TARGETS), score, dtype=np.float32),
        truth=np.zeros((1, TARGETS), dtype=np.float32),
    )
    return path


def test_of(tmp_path: Path) -> None:
    block = BlockStats.of(_block(tmp_path, "oof_fold1_s2026", 0.8))
    assert block.name == "oof_fold1_s2026"
    assert block.predicted.shape == (TARGETS,)
    assert block.predicted.mean() == pytest.approx(0.8, abs=1e-6)
    assert block.spread.mean() == pytest.approx(0.0, abs=1e-6)
    assert block.positive.mean() == pytest.approx(1.0)
    assert block.prevalence.mean() == pytest.approx(0.0)


def test_outside(tmp_path: Path) -> None:
    scatter = (
        BlockStats.of(_block(tmp_path, "oof_fold1_s2026", 0.3)),
        BlockStats.of(_block(tmp_path, "oof_fold2_s2026", 0.5)),
    )
    assert BlockStats.of(_block(tmp_path, "inside", 0.4)).outside(scatter)["mean"] == []
    assert BlockStats.of(_block(tmp_path, "above", 0.9)).outside(scatter)["mean"] == list(range(TARGETS))


def test_line(tmp_path: Path) -> None:
    line = BlockStats.of(_block(tmp_path, "oof_fold1_s2026", 0.8)).line()
    assert "oof_fold1_s2026" in line
    assert "mean 0.8000" in line


def test_axes(tmp_path: Path) -> None:
    axes = BlockStats.of(_block(tmp_path, "oof_fold1_s2026", 0.8)).axes()
    assert set(axes) == {"mean", "calibrated", "spread"}
    assert axes["mean"].mean() == pytest.approx(0.8, abs=1e-6)
    assert axes["spread"].mean() == pytest.approx(0.0, abs=1e-6)
