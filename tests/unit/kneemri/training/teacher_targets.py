"""The teacher loader's one job is alignment, so every test here is about a study getting the wrong row."""

from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.teacher_file import TeacherFile
from kneemri.training.teacher_targets import TeacherTargets


def _written(tmp_path: Path, names: list[str], first_column: list[float]) -> Path:
    """An npz shaped like the producer's, where a study's identity is readable off its first column."""
    scores = np.zeros((len(names), len(TARGETS)), dtype=np.float32)
    scores[:, 0] = first_column
    path = tmp_path / "teacher.npz"
    TeacherFile(path).write(names, scores)
    return path


def test_from_file(tmp_path: Path) -> None:
    path = _written(tmp_path, ["c", "a", "b"], [0.3, 0.1, 0.2])
    teacher = TeacherTargets.from_file(path, ["a", "b", "c"], np.array([0, 1, 2]))
    assert teacher.scores[:, 0].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_from_file_requires_only_the_training_studies(tmp_path: Path) -> None:
    path = _written(tmp_path, ["a", "b"], [0.1, 0.2])
    teacher = TeacherTargets.from_file(path, ["a", "b", "held_out"], np.array([0, 1]))
    assert teacher.scores[:2, 0].tolist() == pytest.approx([0.1, 0.2])
    assert np.isnan(teacher.scores[2]).all()


def test_from_file_raises_when_a_training_study_has_no_reading(tmp_path: Path) -> None:
    path = _written(tmp_path, ["a", "b"], [0.1, 0.2])
    with pytest.raises(KeyError, match="no teacher reading"):
        TeacherTargets.from_file(path, ["a", "b", "missing"], np.array([0, 1, 2]))


def test_from_file_raises_on_the_wrong_number_of_columns(tmp_path: Path) -> None:
    path = tmp_path / "narrow.npz"
    TeacherFile(path).write(["a"], np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="columns"):
        TeacherTargets.from_file(path, ["a"], np.array([0]))


def test_rows(tmp_path: Path) -> None:
    path = _written(tmp_path, ["a", "b", "c"], [0.1, 0.2, 0.3])
    teacher = TeacherTargets.from_file(path, ["a", "b", "c"], np.array([0, 1, 2]))
    assert teacher.rows(np.array([2, 0]))[:, 0].tolist() == pytest.approx([0.3, 0.1])


def test_rows_raises_on_a_study_the_teacher_never_read(tmp_path: Path) -> None:
    path = _written(tmp_path, ["a"], [0.1])
    teacher = TeacherTargets.from_file(path, ["a", "held_out"], np.array([0]))
    with pytest.raises(ValueError, match="no teacher reading"):
        teacher.rows(np.array([0, 1]))
