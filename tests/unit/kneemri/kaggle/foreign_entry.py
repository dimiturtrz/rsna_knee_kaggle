from pathlib import Path

import numpy as np

from core.findings import STUDY, TARGETS
from kneemri.kaggle.foreign_entry import ForeignEntry


def _entry(tmp_path: Path) -> ForeignEntry:
    return ForeignEntry(
        competition=tmp_path,
        kernel=tmp_path / "theirs.py",
        checkpoints=tmp_path / "ckpt",
        series_root=tmp_path / "series",
        working=tmp_path,
    )


def test_ranked() -> None:
    """Five folds become one opinion in RANK space, so a confident fold cannot outvote the order."""
    agree = np.array([[[0.9, 0.1], [0.1, 0.9]]], np.float32)
    loud = np.array([[[0.99, 0.01], [0.01, 0.99]]], np.float32)
    both = ForeignEntry.ranked(np.concatenate([agree, loud]))
    assert both.shape == (2, 2)
    assert np.allclose(both, ForeignEntry.ranked(agree)), "a louder fold that AGREES changes nothing"


def test_table(tmp_path: Path) -> None:
    """The frame carries the study column first and every target, and is held to Submission.check."""
    probabilities = np.tile(np.linspace(0.1, 0.9, len(TARGETS)), (3, 1))
    probabilities[1] = probabilities[1][::-1]
    frame = _entry(tmp_path).table(["a", "b", "c"], probabilities)
    assert list(frame.columns) == [STUDY, *TARGETS]
    assert frame[STUDY].tolist() == ["a", "b", "c"]
