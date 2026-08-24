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
