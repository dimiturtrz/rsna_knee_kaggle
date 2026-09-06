"""The teacher table is a concatenation, so the tests state what may and may not be concatenated."""

from pathlib import Path

import numpy as np
import pytest

from kneemri.evaluation.oof_teacher import OutOfFoldTeacher
from kneemri.teacher_file import TeacherFile

TARGETS = 12
"""Columns a teacher file carries, one per label."""


def _fold(directory: Path, name: str, studies: list[str]) -> Path:
    path = directory / f"{name}.holdout.npz"
    scores = np.full((len(studies), TARGETS), float(len(studies)), dtype=np.float32)
    np.savez(path, studies=np.asarray(studies, dtype="<U64"), predictions=scores)
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
