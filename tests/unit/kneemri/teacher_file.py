"""The npz key names are a FILE FORMAT, so the thing worth asserting is that they have not moved."""

from pathlib import Path

import numpy as np

from kneemri.teacher_file import TeacherFile


def test_write(tmp_path: Path) -> None:
    """A writer and a reader that both spell the names from here agree, which is the module's whole job.

    Pinned by VALUE and not only by identity: a teacher file costs thirty-six minutes of card to write,
    and a rename that compiles would make every one already on disk unreadable in a way whose first
    symptom is a KeyError at the start of a forty-minute train.
    """
    assert (TeacherFile.IDENTIFIERS, TeacherFile.SCORES) == ("identifiers", "scores")

    path = tmp_path / "teacher.npz"
    TeacherFile.write(path, ["s0"], np.zeros((1, 12), dtype=np.float32))
    stored = np.load(path, allow_pickle=True)

    assert stored[TeacherFile.IDENTIFIERS].tolist() == ["s0"]
    assert stored[TeacherFile.SCORES].shape == (1, 12)


def test_merged(tmp_path: Path) -> None:
    """Shards join into one file that carries every study, identifiers and scores staying in step.

    The pass is run as several processes over disjoint strides of the split, so the join is the only
    place a row can lose its study: concatenating the two arrays out of step would produce a file that
    loads, trains, and teaches every student the wrong ordering.
    """
    parts = []
    for shard, studies in enumerate([["s0", "s2"], ["s1"]]):
        part = tmp_path / f"shard{shard}.npz"
        TeacherFile.write(part, studies, np.full((len(studies), 12), float(shard), dtype=np.float32))
        parts.append(part)

    joined = tmp_path / "teacher.npz"
    count = TeacherFile.merged(parts, joined)
    stored = np.load(joined, allow_pickle=True)

    assert count == 3
    assert stored[TeacherFile.IDENTIFIERS].tolist() == ["s0", "s2", "s1"]
    assert stored[TeacherFile.SCORES][:, 0].tolist() == [0.0, 0.0, 1.0]
