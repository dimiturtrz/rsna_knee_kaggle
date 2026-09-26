"""The paired floor's arithmetic, on constructed predictions where the answer is known in advance."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data.labels.label_power import LabelPower
from core.findings import TARGETS
from kneemri.evaluation.paired_floor import PairedFloor
from kneemri.evaluation.scores import ArmPair

STUDIES = 200
RESAMPLES = 60


def _power(tmp_path: Path) -> LabelPower:
    frame = pd.DataFrame({f"{target}__verdict": ["YES"] * 40 + ["NO"] * 60 for target in TARGETS})
    path = tmp_path / "labels.csv"
    frame.to_csv(path, index=False)
    return LabelPower(path)


def _truth() -> np.ndarray:
    """Half positive, half negative on every label, in a fixed alternating pattern — no RNG."""
    column = np.tile([1.0, 0.0], STUDIES // 2)
    return np.column_stack([column] * len(TARGETS)).astype(np.float32)


def _signal() -> np.ndarray:
    """A predictor that is right, so both arms score well above chance and an AUC is defined."""
    truth = _truth().astype(np.float64)
    return truth * 0.6 + 0.2


def test_paired(tmp_path: Path) -> None:
    """Two IDENTICAL arms differ by exactly zero on every resample, so the width is zero."""
    floor = PairedFloor(_power(tmp_path))
    signal = _signal()

    widths = floor.paired(ArmPair(signal, signal, _truth(), RESAMPLES))

    assert set(widths) == set(TARGETS)
    assert widths["MCL"] == pytest.approx(0.0)


def test_unpaired(tmp_path: Path) -> None:
    """The unpaired width comes from the positives actually present, not from a corpus table."""
    widths = PairedFloor(_power(tmp_path)).unpaired(_truth())

    assert widths["MCL"] > 0.0


def test_absolute(tmp_path: Path) -> None:
    """One arm's absolute mean carries a NON-zero width even against itself — nothing pairs it away.

    Scored on an IMPERFECT predictor, deliberately: `_signal` separates the classes exactly, so every
    resample of it scores 1.0 and the width would be zero for want of an error to measure rather than
    because the construction cancels one.
    """
    floor = PairedFloor(_power(tmp_path))
    signal = _signal()
    signal[:40] = 0.5

    width = floor.absolute(signal, _truth(), RESAMPLES)

    assert width > 0.0, "an absolute mean has sampling error the paired construction cannot cancel"
    assert width > floor.paired(ArmPair(signal, signal, _truth(), RESAMPLES))["MCL"]


def test_tightening(tmp_path: Path) -> None:
    floor = PairedFloor(_power(tmp_path))

    ratio = floor.tightening(dict.fromkeys(TARGETS, 0.01), dict.fromkeys(TARGETS, 0.11))

    assert ratio["MCL"] == pytest.approx(11.0)


def test_tightening_of_a_zero_width(tmp_path: Path) -> None:
    """A zero paired width would divide by zero; it reports NaN rather than infinity."""
    ratio = PairedFloor(_power(tmp_path)).tightening(dict.fromkeys(TARGETS, 0.0), dict.fromkeys(TARGETS, 0.1))

    assert np.isnan(ratio["MCL"])


def test_subset(tmp_path: Path) -> None:
    """Without replacement, seeded, and never larger than the corpus it draws from."""
    floor = PairedFloor(_power(tmp_path))

    drawn = floor.subset(50, STUDIES)

    assert len(drawn) == len(set(drawn.tolist())), "without replacement"
    assert np.array_equal(drawn, floor.subset(50, STUDIES)), "seeded, so it is quotable"
    assert len(floor.subset(STUDIES * 2, STUDIES)) == STUDIES, "never more than the corpus holds"


def test_report(tmp_path: Path) -> None:
    """Reports both widths and hands back the paired ones."""
    floor = PairedFloor(_power(tmp_path))
    signal = _signal()

    reported = floor.report(ArmPair(signal, signal, _truth(), RESAMPLES))

    assert reported["MCL"] == pytest.approx(0.0)


def test_mean_width(tmp_path: Path) -> None:
    """Averages over the labels that are defined, ignoring the ones that are not."""
    floor = PairedFloor(_power(tmp_path))

    assert floor.mean_width({"a": 0.02, "b": 0.04, "c": float("nan")}) == pytest.approx(0.03)
