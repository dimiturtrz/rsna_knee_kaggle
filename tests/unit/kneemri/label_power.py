from pathlib import Path

import pandas as pd
import pytest

from kneemri.label_power import LabelPower
from kneemri.model import TARGETS


def _labels(tmp_path: Path, positives: int, studies: int = 100) -> Path:
    verdicts = ["YES"] * positives + ["NO"] * (studies - positives)
    frame = pd.DataFrame({f"{target}__verdict": verdicts for target in TARGETS})
    path = tmp_path / "labels.csv"
    frame.to_csv(path, index=False)
    return path


def test_positives(tmp_path: Path):
    """Counts the YES verdicts only — UNK is silence, not a negative and certainly not a positive."""
    path = _labels(tmp_path, positives=30)
    counted = LabelPower(path).positives()
    assert set(counted) == set(TARGETS)
    assert counted["MCL"] == 30


def test_standard_error(tmp_path: Path):
    """Fewer positives means a wider error bar, and an empty class has no AUC at all.

    The empty case returns NaN rather than 0 on purpose: zero would read as perfect precision, which
    is the most dangerous answer available when a label has nothing to measure.
    """
    power = LabelPower(_labels(tmp_path, positives=30))
    few = power.standard_error(62, 819)
    many = power.standard_error(533, 348)
    assert few > many, "62 positives cannot be measured as precisely as 533"
    assert few == pytest.approx(0.04, abs=0.02), "Fracture-sized counts carry a ~0.04 standard error"
    assert power.standard_error(0, 100) != power.standard_error(0, 100), "an empty class gives NaN"


def test_report(tmp_path: Path):
    """The report leads with the worst-measured label and states the mean's floor."""
    report = LabelPower(_labels(tmp_path, positives=30)).report()
    assert "positives" in report and "SE" in report
    assert "mean of 12 labels" in report
