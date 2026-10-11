"""The npz key names are a FILE FORMAT, so the thing worth asserting is that they have not moved."""

from pathlib import Path

import numpy as np
import pytest

from kneemri.teacher_file import TeacherFile


def test_write(tmp_path: Path) -> None:
    """A writer and a reader that both spell the names from here agree, which is the module's whole job.

    Pinned by VALUE and not only by identity: a teacher file costs thirty-six minutes of card to write,
    and a rename that compiles would make every one already on disk unreadable in a way whose first
    symptom is a KeyError at the start of a forty-minute train.
    """
    assert (TeacherFile.IDENTIFIERS, TeacherFile.SCORES) == ("identifiers", "scores")

    path = tmp_path / "teacher.npz"
    TeacherFile(path).write(["s0"], np.zeros((1, 12), dtype=np.float32))
    stored = np.load(path, allow_pickle=True)

    assert stored[TeacherFile.IDENTIFIERS].tolist() == ["s0"]
    assert stored[TeacherFile.SCORES].shape == (1, 12)


def test_write_embeddings(tmp_path: Path) -> None:
    """Representations land under a name a targets file does not carry, and that is the point.

    Both files are identifiers plus a float matrix, so under one shared name `--teacher-features
    raptor_teacher.npz` would load twelve probabilities as a twelve-wide 'embedding', build a projection
    into it and report a representation arm that had distilled logits. Separate names make that a
    KeyError before the card is touched.
    """
    assert TeacherFile.EMBEDDINGS == "embeddings"

    path = tmp_path / "features.npz"
    TeacherFile(path).write_embeddings(["s0"], np.zeros((1, 7), dtype=np.float32))
    stored = np.load(path, allow_pickle=True)

    assert stored[TeacherFile.EMBEDDINGS].shape == (1, 7)
    assert TeacherFile.SCORES not in stored


def test_aligned() -> None:
    """The corpus's order wins over the file's, and only the TRAINING studies must be present.

    A holdout row that the teacher DID hold would mean a teacher scored on this run's holdout, so it is
    never required; every untaught position is NaN rather than a plausible number, which is what makes a
    fold mismatch surface on the first step instead of as a value the student can quietly learn.
    """
    values = np.array([[3.0], [1.0], [2.0]], dtype=np.float32)
    rows = TeacherFile.aligned(["c", "a", "b"], values, ["a", "b", "c", "held_out"], np.array([0, 1, 2]))

    assert rows[:3, 0].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(rows[3]).all()

    with pytest.raises(KeyError, match="no teacher reading"):
        TeacherFile.aligned(["a"], values[:1], ["a", "missing"], np.array([0, 1]))


def test_merged(tmp_path: Path) -> None:
    """Shards join into one file that carries every study, identifiers and scores staying in step.

    The pass is run as several processes over disjoint strides of the split, so the join is the only
    place a row can lose its study: concatenating the two arrays out of step would produce a file that
    loads, trains, and teaches every student the wrong ordering.
    """
    parts = []
    for shard, studies in enumerate([["s0", "s2"], ["s1"]]):
        part = tmp_path / f"shard{shard}.npz"
        TeacherFile(part).write(studies, np.full((len(studies), 12), float(shard), dtype=np.float32))
        parts.append(part)

    joined = tmp_path / "teacher.npz"
    count = TeacherFile(joined).merged(parts)
    stored = np.load(joined, allow_pickle=True)

    assert count == 3
    assert stored[TeacherFile.IDENTIFIERS].tolist() == ["s0", "s2", "s1"]
    assert stored[TeacherFile.SCORES][:, 0].tolist() == [0.0, 0.0, 1.0]

    # The SAME join over the other format, named at the call: a features merge that wrote a `scores`
    # key would produce a file the feature loader refuses, after the whole decode had been paid for.
    features = [tmp_path / "featshard0.npz", tmp_path / "featshard1.npz"]
    TeacherFile(features[0]).write_embeddings(["s0"], np.zeros((1, 4), dtype=np.float32))
    TeacherFile(features[1]).write_embeddings(["s1"], np.ones((1, 4), dtype=np.float32))
    joined_features = tmp_path / "features.npz"

    assert TeacherFile(joined_features).merged(features, TeacherFile.EMBEDDINGS) == 2
    assert np.load(joined_features)[TeacherFile.EMBEDDINGS].shape == (2, 4)
