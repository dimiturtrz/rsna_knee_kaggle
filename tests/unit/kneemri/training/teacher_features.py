"""The feature loader's one job is alignment, so every test here is about a study getting another's row."""

from pathlib import Path

import numpy as np
import pytest

from kneemri.teacher_file import TeacherFile
from kneemri.training.teacher_features import TeacherFeatures

WIDTH = 5


def _written(tmp_path: Path, names: list[str], first_column: list[float]) -> Path:
    """An npz shaped like the exporter's, where a study's identity is readable off its first column."""
    embeddings = np.zeros((len(names), WIDTH), dtype=np.float32)
    embeddings[:, 0] = first_column
    path = tmp_path / "features.npz"
    TeacherFile.write_embeddings(path, names, embeddings)
    return path


def test_from_file(tmp_path: Path) -> None:
    """The file's rows arrive in the CORPUS's order, not the file's."""
    path = _written(tmp_path, ["c", "a", "b"], [0.3, 0.1, 0.2])
    teacher = TeacherFeatures.from_file(path, ["a", "b", "c"], np.array([0, 1, 2]))

    assert teacher.embeddings[:, 0].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_from_file_refuses_a_targets_file(tmp_path: Path) -> None:
    """A `scores` npz handed to the feature flag must RAISE before the card is touched.

    The two formats are both identifiers plus a float matrix, so under a shared array name this
    invocation would load twelve probabilities as a twelve-wide 'embedding', build a projection into it,
    and report a representation arm that had distilled logits.
    """
    path = tmp_path / "targets.npz"
    TeacherFile.write(path, ["a"], np.zeros((1, 12), dtype=np.float32))

    with pytest.raises(KeyError):
        TeacherFeatures.from_file(path, ["a"], np.array([0]))


def test_width(tmp_path: Path) -> None:
    """The teacher's width is READ off the file, because it is whatever trunk wrote it.

    It is the target side of the projection the objective builds, so a width taken from a default rather
    than from the file would size a map onto a teacher that is not this one.
    """
    path = _written(tmp_path, ["a"], [0.1])

    assert TeacherFeatures.from_file(path, ["a"], np.array([0])).width() == WIDTH


def test_rows(tmp_path: Path) -> None:
    """A batch's vectors come back in the batch's own order."""
    path = _written(tmp_path, ["a", "b", "c"], [0.1, 0.2, 0.3])
    teacher = TeacherFeatures.from_file(path, ["a", "b", "c"], np.array([0, 1, 2]))

    assert teacher.rows(np.array([2, 0]))[:, 0].tolist() == pytest.approx([0.3, 0.1])


def test_rows_raises_on_a_study_the_teacher_never_read(tmp_path: Path) -> None:
    """An untaught study reaching a batch is a loud failure, not a cosine descended against NaN."""
    path = _written(tmp_path, ["a"], [0.1])
    teacher = TeacherFeatures.from_file(path, ["a", "held_out"], np.array([0]))

    with pytest.raises(ValueError, match="no teacher reading"):
        teacher.rows(np.array([0, 1]))
