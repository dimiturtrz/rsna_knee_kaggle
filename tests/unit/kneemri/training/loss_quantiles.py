"""The five quantiles an epoch record carries, over known values and over the empty epoch."""

import math

import numpy as np
import pytest

from kneemri.training.loss_quantiles import LOSS_QUANTILES, EpochLosses

HUNDRED = 100
HALF = 50


def test_collect():
    """Each batch is kept whole, so the epoch's shape is read off every study that contributed."""
    losses = EpochLosses()
    losses.collect(np.zeros(3, dtype=np.float64))
    losses.collect(np.ones(4, dtype=np.float64))

    assert [len(batch) for batch in losses.batches] == [3, 4]


def test_quantiles():
    """Over 0..99 in two batches the five levels are the values the levels name."""
    values = np.arange(HUNDRED, dtype=np.float64)
    losses = EpochLosses()
    losses.collect(values[:HALF])
    losses.collect(values[HALF:])

    quantiles = losses.quantiles()

    assert tuple(quantiles) == LOSS_QUANTILES, "the record's key shape is the module's own tuple"
    assert quantiles["q50"] == pytest.approx(np.quantile(values, 0.50))
    assert quantiles["q95"] == pytest.approx(np.quantile(values, 0.95))
    assert quantiles["q25"] < quantiles["q75"] < quantiles["q95"]


def test_quantiles_of_an_epoch_no_study_reached():
    """Nothing collected and an empty batch both read NaN under every key, never a missing key."""
    empty = EpochLosses()
    emptied = EpochLosses()
    emptied.collect(np.zeros(0, dtype=np.float64))

    for losses in (empty, emptied):
        quantiles = losses.quantiles()

        assert tuple(quantiles) == LOSS_QUANTILES
        assert all(math.isnan(value) for value in quantiles.values())
