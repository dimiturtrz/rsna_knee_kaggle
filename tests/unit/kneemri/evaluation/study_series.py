"""The mirror for the acquisition table: every public method called, and asserted."""

from pathlib import Path

import pandas as pd

from kneemri.evaluation.study_series import StudySeries

STUDIES = ["s0", "s1", "s2", "s3", "s4", "s5"]


def _series(tmp_path: Path) -> StudySeries:
    rows = [
        {"StudyInstanceUID": study, "Fluid_Sensitive": int(index % 2), "Fat_Suppression": int(index % 2)}
        for study in STUDIES
        for index in range(4 + STUDIES.index(study) % 3)
    ]
    path = tmp_path / "train_series.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return StudySeries(path)


def test_counts(tmp_path: Path) -> None:
    counts = _series(tmp_path).counts(STUDIES)
    assert counts.tolist() == [4, 5, 6, 4, 5, 6]


def test_fluid(tmp_path: Path) -> None:
    fluid = _series(tmp_path).fluid(STUDIES)
    assert fluid.tolist() == [2, 2, 3, 2, 2, 3]


def test_redundant(tmp_path: Path) -> None:
    assert _series(tmp_path).redundant()
