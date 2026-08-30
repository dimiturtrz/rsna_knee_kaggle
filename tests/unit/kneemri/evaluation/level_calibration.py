"""The map from a reader's levels to their measured rate, and the pooling that keeps it monotone."""

import numpy as np

from kneemri.evaluation.level_calibration import LevelCalibration, LevelRates

LEVELS = np.array([0.08, 0.28, 0.68, 0.94], dtype=np.float64)
"""Four of the five levels the reader emits, enough to carry an inversion and both ends."""


VOCABULARY = np.array([0.08, 0.28, 0.68, 0.94], dtype=np.float64)
"""What the reader emits — deliberately without 0.5, which is `GoldRows`' unread fill."""


def test_vocabulary() -> None:
    """The reader's levels come from the whole table, and a missing cell is not one of them."""
    table = np.array([[0.08, 0.94], [0.68, np.nan], [0.08, 0.68]], dtype=np.float32)

    assert list(LevelCalibration().vocabulary(table)) == [0.08, 0.68, 0.94]


def test_observed() -> None:
    """The rate is pooled over findings, counts cells rather than studies, and drops unread ones."""
    reader = np.array([[0.08, 0.94], [0.08, 0.94], [0.68, 0.68]], dtype=np.float32)
    gold = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    rates = LevelCalibration().observed(reader, gold, VOCABULARY)

    assert list(rates.levels) == [0.08, 0.68, 0.94]
    assert list(rates.counts) == [2.0, 2.0, 2.0], "six cells over three studies, not three"
    assert list(rates.raw) == [0.0, 0.5, 1.0]
    assert list(rates.fitted) == [0.0, 0.5, 1.0], "already monotone, so nothing pools"

    # The unread fill sits at 0.5 and is outside the vocabulary, so it informs nothing.
    unread = np.array([[0.08, 0.5], [0.08, 0.5], [0.68, 0.94]], dtype=np.float32)
    dropped = LevelCalibration().observed(unread, gold, VOCABULARY)
    assert list(dropped.levels) == [0.08, 0.68, 0.94]
    assert int(dropped.counts.sum()) == 4, "two of six cells were never read"


def test_monotone() -> None:
    """A violated pair becomes their weighted mean, and the thin bucket is the one that moves."""
    calibration = LevelCalibration()

    # The shape off disk: level 0.50 reads 0.417 on twelve cells against 0.68's 0.378 on ninety-eight.
    fitted = calibration.monotone(
        np.array([0.041, 0.136, 0.417, 0.378, 0.688, 0.958], dtype=np.float64),
        np.array([196.0, 177.0, 12.0, 98.0, 141.0, 72.0], dtype=np.float64),
    )

    assert len(fitted) == 6, "a pooled block is spread back over the levels it covered"
    assert list(fitted) == sorted(fitted), "the map a confidence deserves cannot decrease"
    assert fitted[2] == fitted[3], "the violating pair share one value"
    pooled = (0.417 * 12 + 0.378 * 98) / 110
    assert abs(fitted[2] - pooled) < 1e-9
    assert abs(pooled - 0.378) < abs(pooled - 0.417), "the twelve-cell bucket moves, not the ninety-eight"

    # Untouched where it is already non-decreasing.
    unchanged = calibration.monotone(np.array([0.1, 0.5, 0.9]), np.array([10.0, 10.0, 10.0]))
    assert list(unchanged) == [0.1, 0.5, 0.9]


def test_applied() -> None:
    """Every fitted level lands on its cells, and a level the fit never saw is left alone."""
    rates = LevelRates(
        levels=LEVELS,
        counts=np.array([1.0, 1.0, 1.0, 1.0]),
        raw=np.array([0.04, 0.14, 0.38, 0.96]),
        fitted=np.array([0.04, 0.14, 0.38, 0.96]),
    )

    mapped = LevelCalibration().applied(np.array([[0.08, 0.68], [0.94, 0.5]], dtype=np.float32), rates)

    assert np.allclose(mapped[0], [0.04, 0.38])
    assert mapped[1][0] == np.float32(0.96)
    assert mapped[1][1] == np.float32(0.5), "an unseen level is not invented a value"


def test_line() -> None:
    """The row carries the count and both rates, because the pooling is the evidence."""
    rates = LevelRates(
        levels=np.array([0.5, 0.68]),
        counts=np.array([12.0, 98.0]),
        raw=np.array([0.417, 0.378]),
        fitted=np.array([0.382, 0.382]),
    )

    line = rates.line()

    assert "n=  12" in line
    assert "raw 0.417" in line
    assert "fitted 0.382" in line
