"""The rotation assembler refuses the one defect that does not announce itself: a study held out twice."""

from pathlib import Path

import numpy as np
import pytest

from kneemri.runs.fold_rotation import COMPOSITE_SOURCE, FoldCoverage, OutOfFoldRotation
from kneemri.runs.holdout_record import HoldoutReading, HoldoutRecord

LABELS = 12
CORPUS = 6


def _reading(names: list[str], seed: int) -> HoldoutReading:
    generator = np.random.default_rng(seed)
    return HoldoutReading(
        studies=names,
        predictions=generator.random((len(names), LABELS)),
        truth=generator.integers(0, 2, (len(names), LABELS)).astype(np.float32),
        epoch=seed,
        source="trained",
    )


def test_overlapping() -> None:
    """Names every study two folds both claim, and nothing when the rotation is disjoint."""
    disjoint = OutOfFoldRotation([_reading(["a", "b"], 1), _reading(["c", "d"], 2)])
    assert disjoint.overlapping() == []

    shared = OutOfFoldRotation([_reading(["a", "b"], 1), _reading(["b", "c"], 2)])
    assert shared.overlapping() == ["b"]


def test_labels() -> None:
    """Agrees on the label count, and refuses two folds graded by rulers of different widths."""
    rotation = OutOfFoldRotation([_reading(["a"], 1), _reading(["b"], 2)])
    assert rotation.labels() == LABELS

    narrow = _reading(["c"], 3)
    mismatched = OutOfFoldRotation(
        [
            _reading(["a"], 1),
            HoldoutReading(
                studies=narrow.studies,
                predictions=narrow.predictions[:, :2],
                truth=narrow.truth[:, :2],
                epoch=narrow.epoch,
                source=narrow.source,
            ),
        ]
    )
    with pytest.raises(ValueError, match="different rulers"):
        mismatched.labels()


def test_assemble() -> None:
    """Concatenates disjoint folds into one row per study, and refuses an overlapping rotation."""
    rotation = OutOfFoldRotation([_reading(["a", "b"], 1), _reading(["c", "d", "e"], 2)])
    studies, predictions, truth, coverage = rotation.assemble(CORPUS)
    assert studies == ["a", "b", "c", "d", "e"]
    assert predictions.shape == (5, LABELS)
    assert truth.shape == (5, LABELS)
    assert coverage.studies == 5

    with pytest.raises(ValueError, match="TRAINED on them"):
        OutOfFoldRotation([_reading(["a"], 1), _reading(["a"], 2)]).assemble(CORPUS)


def test_write(tmp_path: Path) -> None:
    """Writes an ordinary holdout record every existing scorer can read, stamped as a composite."""
    rotation = OutOfFoldRotation([_reading(["a", "b"], 1), _reading(["c", "d"], 2)])
    coverage = rotation.write(tmp_path / "rot.pt", CORPUS)

    reading = HoldoutRecord(tmp_path / "rot.pt").read()
    assert reading is not None
    assert list(reading.studies) == ["a", "b", "c", "d"]
    assert reading.source == COMPOSITE_SOURCE
    assert coverage.folds == 2


def test_share() -> None:
    """The denominator as a number, and zero rather than a crash on an empty corpus."""
    assert FoldCoverage(studies=2204, folds=2, corpus=4407).share() == pytest.approx(0.5001, abs=1e-3)
    assert FoldCoverage(studies=0, folds=0, corpus=0).share() == 0.0


def test_complete() -> None:
    """True only when every corpus study got exactly one out-of-fold prediction."""
    assert FoldCoverage(studies=4407, folds=3, corpus=4407).complete()
    assert not FoldCoverage(studies=3498, folds=2, corpus=4407).complete()
