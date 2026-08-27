"""The per-finding offsets, and the null they must return when the findings really do share one scale.

The estimator's whole job is to tell a real between-finding difference from one manufactured by 57
Bernoulli draws, so the two tests that matter are the two answers: cells generated from a genuinely
shifted finding must recover its sign, and cells generated from ONE shared scale must shrink to zero.
"""

import numpy as np
import pandas as pd
import pytest

from core.data.labels.finding_calibration import POOLED, FindingCalibration
from core.data.labels.targets import MEASURED_AGREEMENT, Supervision
from core.findings import TARGETS

MILD = 0.68
"""The level whose pooled worth (0.378) sits nearest the middle, so a shift moves it visibly."""

CELLS_PER_FINDING = 200
"""Comfortably more than the 57 the real table carries — this test is about the estimator, not the corpus."""

POOLED_RATE = 0.378
"""What `MEASURED_AGREEMENT` says the mild level is worth, pooled — the null these tests sit at."""


def _cells(rates: dict[str, float]) -> pd.DataFrame:
    """Cells for each finding, all written at one level, positive at EXACTLY that finding's rate.

    The counts are exact rather than drawn, so the estimator is being asked about a rate it was
    literally handed. A sampled fixture would make the null test a coin flip on the seed: at a true
    between-finding variance of zero the moment estimator is negative only half the time.
    """
    rows = []
    for target, rate in rates.items():
        positives = round(rate * CELLS_PER_FINDING)
        rows.extend((f"s{index}", target, MILD, float(index < positives)) for index in range(CELLS_PER_FINDING))
    return pd.DataFrame(rows, columns=["study", "target", "value", "truth"])


def test_expected() -> None:
    """Each level is worth what it was measured at, and an unknown value passes through as itself."""
    values = np.array([0.68, 0.94, 0.5], dtype=np.float64)

    worth = FindingCalibration().expected(values)

    assert worth[0] == 0.378, "the mild level trains at the rate the radiologist agreed at"
    assert worth[1] == 0.958
    assert worth[2] == 0.5, "a value the reader does not emit is not snapped to a nearest level"


def test_offset() -> None:
    """The fitted shift moves the pooled rate onto the rate actually observed."""
    expected = np.full(100, 0.378)
    truth = np.concatenate([np.ones(70), np.zeros(30)])

    shift = FindingCalibration().offset(expected, truth)

    moved = 1.0 / (1.0 + np.exp(-(np.log(0.378 / 0.622) + shift)))
    assert moved == pytest.approx(0.70, abs=1e-6), "70 positives of 100 is what the shifted rate reproduces"


def test_information() -> None:
    """Information grows with the cells behind a shift — it is the inverse of its sampling variance."""
    calibration = FindingCalibration()
    few = calibration.information(np.full(10, 0.378), POOLED)
    many = calibration.information(np.full(100, 0.378), POOLED)

    assert many > few
    assert np.isclose(many, 10 * few), "each cell contributes the same p(1-p) at the same fitted value"


def test_fitted() -> None:
    """A finding the reader over-calls fits a negative shift; one it misses fits a positive one."""
    fitted = FindingCalibration().fitted(_cells({"over": 0.10, "under": 0.90}))
    raw = dict(zip(fitted["target"], fitted["raw"], strict=True))
    cells = dict(zip(fitted["target"], fitted["cells"], strict=True))

    assert float(raw["over"]) < 0.0
    assert float(raw["under"]) > 0.0
    assert int(cells["over"]) == CELLS_PER_FINDING


def test_spread() -> None:
    """The between-finding variance is what survives subtracting the sampling variance, never less than zero."""
    calibration = FindingCalibration()

    apart = calibration.spread(calibration.fitted(_cells({"over": 0.10, "under": 0.90})))
    together = calibration.spread(calibration.fitted(_shared()))

    assert apart > 0.0, "two findings 0.8 apart in rate are not one scale"
    assert together == 0.0, "three findings drawn from ONE rate must not manufacture a spread"


def _shared() -> pd.DataFrame:
    """Three findings sitting at the SAME rate — the null the estimator has to be able to return."""
    return _cells(dict.fromkeys(["a", "b", "c"], POOLED_RATE))


def test_offsets() -> None:
    """The shipped table is the raw shift drawn toward pooled, and it is exactly pooled when there is no spread."""
    calibration = FindingCalibration()

    apart = calibration.offsets(_cells({"over": 0.10, "under": 0.90}))
    together = calibration.offsets(_shared())

    assert abs(apart["over"]) > 0.5 and apart["over"] < 0.0, "plenty of cells, so little is shrunk away"
    assert set(together.values()) == {POOLED}, "no spread returns every finding to today's table"


def test_shifted() -> None:
    """A cell's calibrated value moves by its own finding's offset, and by nothing when it has none."""
    values = np.array([MILD, MILD], dtype=np.float64)

    moved = FindingCalibration().shifted(values, ["raised", "absent"], {"raised": 1.0})

    assert moved[0] > 0.378, "a positive offset raises what the cell trains at"
    assert np.isclose(moved[1], 0.378), "a finding not in the table keeps the pooled value"


def test_valued() -> None:
    """A cell is given the LEVEL the reader wrote, joined from the label table rather than from its verdict."""
    cells = pd.DataFrame([("s0", "ACL", "YES", 1.0)], columns=["study", "target", "verdict", "truth"])
    read = pd.DataFrame([{"StudyInstanceUID": "s0", **dict.fromkeys(TARGETS, MILD), "ACL": 0.94}])

    valued = FindingCalibration.valued(cells, read)

    assert valued.loc[0, "value"] == 0.94, "the severe level, which the YES token could not have told apart"


def test_report() -> None:
    """The report names the spread, every finding's shift, and what a mid-scale cell becomes under it."""
    lines = FindingCalibration().report(_cells({"over": 0.10, "under": 0.90}))

    text = "\n".join(lines)
    assert "between-finding variance" in text
    assert "over" in text and "under" in text


def test_report_says_so_when_the_spread_dies() -> None:
    """The null is stated in words, not left as a column of zeroes for a reader to notice."""
    lines = FindingCalibration().report(_shared())

    assert any("SPREAD DID NOT SURVIVE" in line for line in lines)


def test_offsets_are_fitted_against_the_shipped_level_map() -> None:
    """The estimator's default supervision is the calibrated one the offsets are a correction TO."""
    assert FindingCalibration().supervision == Supervision(calibrated=True)
    assert FindingCalibration().supervision.measured == MEASURED_AGREEMENT
