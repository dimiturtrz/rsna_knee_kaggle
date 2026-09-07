"""The witness is a residual correlation, so the tests state what the truth is supposed to remove."""

from pathlib import Path

import numpy as np
import pytest

from kneemri.evaluation.teacher_witness import FoldReadings, TeacherWitness

STUDIES = 64
"""Rows in each synthetic reading — enough that a correlation is not a two-point line."""

LABELS = 3
"""Columns, one per finding, since every number here is a mean over labels."""


def _truth() -> np.ndarray:
    return (np.arange(STUDIES * LABELS).reshape(STUDIES, LABELS) % 2).astype(np.float64)


def test_agreement() -> None:
    truth = _truth()
    noise = np.linspace(0.0, 1.0, STUDIES * LABELS).reshape(STUDIES, LABELS)
    assert TeacherWitness(truth).agreement(noise, noise) == 1.0


def test_mimicry() -> None:
    truth = _truth()
    rng = np.random.default_rng(0)
    shared = rng.normal(size=(STUDIES, LABELS))
    apart = rng.normal(size=(STUDIES, LABELS))
    witness = TeacherWitness(truth)
    # Two readings that agree ONLY through the truth: their probabilities correlate, their residuals do not.
    assert witness.mimicry(truth + shared, truth + apart) < 0.5
    assert witness.mimicry(truth + shared, truth + shared) == 1.0


def test_lines() -> None:
    truth = _truth()
    reading = truth * 0.9 + 0.05
    lines = TeacherWitness(truth).lines({"student": reading}, {"leaky": reading})
    assert any("macro AUC" in line for line in lines)
    assert any("student" in line and "leaky" in line and "mimicry" in line for line in lines)


def test_aligned(tmp_path: Path) -> None:
    """A record is joined onto the card's study order by NAME, never by the order it was saved in."""
    record = tmp_path / "run.holdout.npz"
    np.savez(
        record,
        studies=np.array(["b", "a", "c"]),
        predictions=np.array([[0.2], [0.1], [0.3]]),
        truth=np.array([[1.0], [0.0], [1.0]]),
    )
    predictions, truth = FoldReadings.aligned(record, ["a", "c", "b"])
    assert predictions[:, 0].tolist() == [0.1, 0.3, 0.2]
    assert truth[:, 0].tolist() == [0.0, 1.0, 1.0]


def test_mean() -> None:
    """A teacher set's mean is taken over the members it named, and a missing one refuses."""
    read = [(Path("one.pt"), np.zeros((2, 1))), (Path("two.pt"), np.ones((2, 1)))]
    assert FoldReadings.mean(read, (Path("one.pt"), Path("two.pt")))[0, 0] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="was skipped"):
        FoldReadings.mean(read, (Path("one.pt"), Path("missing.pt")))
