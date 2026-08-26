"""Two readers whose answers are known by construction: one perfect, one inverted, one an exact copy."""

from pathlib import Path

import numpy as np
import pandas as pd

from core.findings import TARGETS
from kneemri.evaluation.table_separation import TableSeparation

STUDIES = 12
DRAWS = 60
"""Enough draws for a percentile, few enough that the suite stays a unit test."""


def _truth(index: int, column: int) -> float:
    """A checkerboard, so every finding carries six positives and six negatives and none is degenerate."""
    return float((index + column) % 2)


def _table(path: Path, *, inverted: bool) -> Path:
    """One reader's csv over the same studies, either reading the truth or reading it backwards."""
    rows = [
        {"StudyInstanceUID": f"s{index}"}
        | {
            target: (1.0 - _truth(index, column)) if inverted else _truth(index, column)
            for column, target in enumerate(TARGETS)
        }
        for index in range(STUDIES)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _train(path: Path) -> Path:
    """A train.csv annotating every study on every finding, which is what `annotated` keeps."""
    return _table(path, inverted=False)


def test_readings(tmp_path: Path):
    """Every table comes back in the annotations' row order, with all twelve columns."""
    annotations, readings = TableSeparation().readings(
        _train(tmp_path / "train.csv"),
        {"perfect": _table(tmp_path / "perfect.csv", inverted=False)},
    )

    assert annotations.shape == (STUDIES, len(TARGETS))
    assert readings["perfect"].shape == annotations.shape
    assert np.allclose(readings["perfect"], annotations), "the table read the truth, so it must equal it"


def test_mean(tmp_path: Path):
    """A reader that reproduces the annotations scores 1.0; one that inverts them scores 0.0."""
    annotations, readings = TableSeparation().readings(
        _train(tmp_path / "train.csv"),
        {
            "perfect": _table(tmp_path / "perfect.csv", inverted=False),
            "inverted": _table(tmp_path / "inverted.csv", inverted=True),
        },
    )
    separation = TableSeparation()

    assert separation.mean(readings["perfect"], annotations) == 1.0
    assert separation.mean(readings["inverted"], annotations) == 0.0


def test_separation(tmp_path: Path):
    """An opposite reader is separated from a perfect one; an identical copy is not."""
    separation = TableSeparation()
    annotations, readings = separation.readings(
        _train(tmp_path / "train.csv"),
        {
            "perfect": _table(tmp_path / "perfect.csv", inverted=False),
            "inverted": _table(tmp_path / "inverted.csv", inverted=True),
            "copy": _table(tmp_path / "copy.csv", inverted=False),
        },
    )

    point, (floor, ceiling) = separation.separation(
        readings["perfect"], readings["inverted"], annotations, resamples=DRAWS
    )
    assert point == -1.0
    assert ceiling < 0.0, "a reader that inverts every finding is resolvable on any draw"

    same, (low, high) = separation.separation(readings["perfect"], readings["copy"], annotations, resamples=DRAWS)
    assert (same, low, high) == (0.0, 0.0, 0.0), "the same table twice differs by nothing on every draw"


def test_report(tmp_path: Path):
    """Every table gets a mean line and every pair a verdict, and the verdict follows the interval."""
    separation = TableSeparation()
    annotations, readings = separation.readings(
        _train(tmp_path / "train.csv"),
        {
            "perfect": _table(tmp_path / "perfect.csv", inverted=False),
            "inverted": _table(tmp_path / "inverted.csv", inverted=True),
        },
    )

    lines = separation.report(annotations, readings, resamples=DRAWS)

    assert f"{STUDIES} annotated studies" in lines[0]
    assert any(line.startswith("perfect") and "1.0000" in line for line in lines)
    assert any("inverted - perfect" in line and line.endswith("SEPARATED") for line in lines)
