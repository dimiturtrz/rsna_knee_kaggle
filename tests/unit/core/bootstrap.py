"""The shared resampling protocol: does an interval bracket the mean, and does one stream stay one stream?"""

import numpy as np
import pandas as pd

from core.bootstrap import (
    INTERVAL,
    RESAMPLE_SEED,
    Bootstrap,
    StudyBootstrap,
    StudyFrameBootstrap,
)

DRAWS = 40
"""Enough draws that a percentile exists, few enough that the suite stays a unit test."""

STUDIES = 25


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


def test_draws():
    """Whole studies, with replacement, in range — and the same sequence every time it is asked.

    The seeding is the load-bearing part. Three modules had this loop written out, and the refactor that
    merged them is only safe because the draw ORDER is identical: every interval already published would
    otherwise move without anything reporting that it had.
    """
    bootstrap = StudyBootstrap(resamples=DRAWS)

    drawn = list(bootstrap.draws(STUDIES))

    assert len(drawn) == DRAWS
    assert all(index.shape == (STUDIES,) for index in drawn)
    assert all(int(index.min()) >= 0 and int(index.max()) < STUDIES for index in drawn)
    assert any(len(np.unique(index)) < STUDIES for index in drawn), "with replacement, not a permutation"
    assert [index.tolist() for index in bootstrap.draws(STUDIES)] == [index.tolist() for index in drawn]


def test_study_bootstrap_interval():
    """Brackets the quantity measured on the full sample, and is reproducible."""
    values = np.linspace(0.0, 1.0, STUDIES)
    bootstrap = StudyBootstrap(resamples=DRAWS)

    low, high = bootstrap.interval(STUDIES, lambda drawn: float(values[drawn].mean()))

    assert low < float(values.mean()) < high
    assert bootstrap.interval(STUDIES, lambda drawn: float(values[drawn].mean())) == (low, high)


def test_per_label():
    """Every label measured gets a bound, INCLUDING one that was undefined on every draw.

    The missing-key case is the one the hand-written copies got right by pre-seeding their dict and a
    naive merge would get wrong: callers index this result by label, so dropping an all-NaN label turns a
    printable "undefined" into a KeyError.
    """
    values = np.linspace(0.0, 1.0, STUDIES)

    bounds = StudyBootstrap(resamples=DRAWS).per_label(
        STUDIES,
        lambda drawn: {"defined": float(values[drawn].mean()), "undefined": float("nan")},
    )

    assert set(bounds) == {"defined", "undefined"}
    assert bounds["defined"][0] < float(values.mean()) < bounds["defined"][1]
    assert all(np.isnan(bound) for bound in bounds["undefined"])


def test_percentile_interval():
    """The two-sided pair off a finished list, and NaN rather than a raise when nothing was defined."""
    assert all(np.isnan(bound) for bound in Bootstrap.percentile_interval([]))

    spread = [float(value) for value in np.linspace(0.0, 100.0, 101)]
    low, high = Bootstrap.percentile_interval(spread)

    assert (round(low, 6), round(high, 6)) == (2.5, 97.5), "the pair is INTERVAL, read off the values"
    assert Bootstrap.percentile_interval(spread, (5.0, 95.0)) == (5.0, 95.0), "a caller's own band is honoured"


def _frame() -> pd.DataFrame:
    generator = np.random.default_rng(7)
    studies = np.repeat([f"s{index}" for index in range(STUDIES)], 3)
    return pd.DataFrame({"study": studies, "value": generator.random(len(studies))})


def test_study_frame_bootstrap_interval():
    """Whole studies drawn by name, a None draw skipped, and no defined draw reads (nan, nan)."""
    cells = _frame()
    bootstrap = StudyFrameBootstrap(resamples=DRAWS)

    low, high = bootstrap.interval(cells, lambda sample: float(sample["value"].mean()))

    assert low < float(cells["value"].mean()) < high
    assert bootstrap.interval(cells, lambda sample: float(sample["value"].mean())) == (low, high)
    skipping = bootstrap.interval(cells, lambda sample: None if sample["value"].mean() > high else 0.0)
    assert skipping == (0.0, 0.0)
    assert all(np.isnan(bound) for bound in bootstrap.interval(cells, lambda _sample: None))
