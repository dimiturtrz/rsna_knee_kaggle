"""The mirror for the population cut: every public method called, and asserted."""

from pathlib import Path

import numpy as np
import pandas as pd

from core.findings import TARGETS
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.population_split import SERIES_BANDS, PopulationRow, PopulationSplit
from kneemri.evaluation.residual_split import ResidualSplit
from kneemri.evaluation.study_series import StudySeries

STUDIES = ["s0", "s1", "s2", "s3", "s4", "s5"]


def _row(errors: int = 10, unanimous: int = 4) -> PopulationRow:
    return PopulationRow(stratum="5 series", studies=5, positives=20, errors=errors, unanimous=unanimous)


def _split() -> PopulationSplit:
    truth = np.zeros((len(STUDIES), len(TARGETS)))
    truth[:3] = 1.0
    beliefs = np.tile(np.linspace(1.0, 0.0, len(STUDIES))[:, np.newaxis], (1, len(TARGETS)))
    predictions = [beliefs, beliefs[::-1].copy(), beliefs]
    pool = Pool(names=["a", "b", "c"], predictions=predictions, truth=truth)
    return PopulationSplit(ResidualSplit(pool), STUDIES)


def _series(tmp_path: Path) -> StudySeries:
    rows = [
        {"StudyInstanceUID": study, "Fluid_Sensitive": int(index % 2), "Fat_Suppression": int(index % 2)}
        for study in STUDIES
        for index in range(4 + STUDIES.index(study) % 3)
    ]
    path = tmp_path / "train_series.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return StudySeries(path)


def test_per_study() -> None:
    assert _row(errors=10).per_study == 2.0


def test_per_positive() -> None:
    assert _row(errors=10).per_positive == 0.5


def test_unanimous_share() -> None:
    assert _row(errors=10, unanimous=5).unanimous_share == 0.5


def test_line() -> None:
    line = _row().line()
    assert "5 series" in line
    assert "40.0%" in line


def test_per_study_errors() -> None:
    errors = _split().per_study_errors()
    assert errors.shape == (len(STUDIES),)
    assert int(errors.sum()) >= 0


def test_per_study_unanimous() -> None:
    unanimous = _split().per_study_unanimous()
    assert unanimous.shape == (len(STUDIES),)
    assert (unanimous <= len(TARGETS)).all()


def test_banded() -> None:
    values = np.array([4, 5, 6, 4, 5, 7], dtype=np.int64)
    rows = _split().banded(values, SERIES_BANDS)
    assert len(rows) == len(SERIES_BANDS)
    assert [row.studies for row in rows] == [2, 2, 1, 1]


def test_report(tmp_path: Path) -> None:
    lines = _split().report(_series(tmp_path))
    assert "by series count" in lines
    assert any("fluid-sensitive" in line for line in lines)
