"""The mirror for the slice readouts: each reduction called on a stack built to separate them."""

import numpy as np

from kneemri.evaluation.slice_readout import READOUTS, SliceMax, SliceMean, SliceMoments

PEAK = 5.0
"""The one slice that stands above a flat stack, so max and mean cannot agree."""


def _stack() -> np.ndarray:
    """One study whose channel varies along the stack, against one that is flat there."""
    array = np.ones((2, 2, 4, 3), dtype=np.float16)
    array[0, :, 3, :] = PEAK
    return array


def test_slice_mean_reduce() -> None:
    """The average over both axes, which is the reduction every ledger number was read with."""
    reduced = SliceMean().reduce(_stack())
    assert reduced.shape == (2, 3)
    assert reduced[0, 0] == 2.0
    assert reduced[1, 0] == 1.0


def test_slice_max_reduce() -> None:
    """The peak along the stack survives, which is what the mean averages away."""
    reduced = SliceMax().reduce(_stack())
    assert reduced.shape == (2, 3)
    assert reduced[0, 0] == PEAK
    assert reduced[1, 0] == 1.0


def test_slice_moments_reduce() -> None:
    """Twice the width: the average, then the spread along the stack that distinguishes the two."""
    reduced = SliceMoments().reduce(_stack())
    assert reduced.shape == (2, 6)
    assert reduced[0, 0] == 2.0
    assert reduced[0, 3] > 0.0
    assert reduced[1, 3] == 0.0


def test_readouts_are_keyed_by_name() -> None:
    """The keys are the values `--slice-readout` takes, so a typo cannot silently pick the default."""
    assert sorted(READOUTS) == ["max", "mean", "moments"]
    assert READOUTS["mean"].name == "mean"
