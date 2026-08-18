"""The shared resampling protocol: does an interval bracket the mean, and does one stream stay one stream?"""

import numpy as np

from core.bootstrap import INTERVAL, RESAMPLE_SEED, Bootstrap


def test_mean_interval():
    """The interval brackets the sample mean and narrows as the sample grows — the two things it claims."""
    values = np.linspace(0.0, 1.0, 200)

    low, high = Bootstrap.seeded().mean_interval(values)

    assert low < float(values.mean()) < high
    wide_low, wide_high = Bootstrap.seeded().mean_interval(np.linspace(0.0, 1.0, 20))
    assert (high - low) < (wide_high - wide_low), "200 samples must buy a tighter bound than 20"


def test_seeded():
    """One object is one stream: a second quantity continues it rather than redrawing the first's.

    This is the reason the protocol is an object and not a function taking a seed. Restarting per
    quantity would give two measurements in the same report identical draws, so their intervals would be
    correlated by construction and neither would be the independent check it is quoted as.
    """
    bootstrap = Bootstrap.seeded()
    values = np.linspace(0.0, 1.0, 50)

    first = bootstrap.mean_interval(values)
    second = bootstrap.mean_interval(values)

    assert first != second, "the second call restarted the stream"
    assert Bootstrap.seeded(RESAMPLE_SEED).mean_interval(values) == first, "and it is still seeded"


def test_interval_is_a_two_sided_95_percent_pair():
    """Legislated, not read off the tree: a report that widened its own bound would be unfalsifiable."""
    low, high = INTERVAL

    assert (low, high) == (2.5, 97.5)
    assert high - low == 95.0
