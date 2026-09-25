"""What a learned combiner is allowed to claim, asked of the arithmetic rather than of a pool on disk."""

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.stack_combiner import PENALTIES, RidgeFit, StackCombiner

STUDIES = 60
MEMBERS = 4


def _truth() -> np.ndarray:
    """A deterministic label table where every finding has both classes present."""
    rows = np.arange(STUDIES)[:, None]
    columns = np.arange(len(TARGETS))[None, :]
    return ((rows + columns) % 3 == 0).astype(np.float64)


def _pool(truth: np.ndarray | None = None) -> Pool:
    """Four members that see the truth through different deterministic distortions."""
    table = _truth() if truth is None else truth
    rows = np.arange(STUDIES)[:, None]
    columns = np.arange(len(TARGETS))[None, :]
    predictions = [table + 0.1 * (((rows * 7 + member * 13 + columns * 5) % 10) / 10.0) for member in range(MEMBERS)]
    return Pool([f"m{member}" for member in range(MEMBERS)], predictions, table)


def test_applied():
    """A fit scores a row by its centred features, which is the only thing an AUC then reads."""
    fit = RidgeFit(np.array([1.0, 2.0]), np.array([3.0, -1.0]))

    scored = fit.applied(np.array([[1.0, 2.0], [2.0, 2.0], [1.0, 3.0]]))

    assert scored[0] == pytest.approx(0.0), "a row AT the centre carries no evidence either way"
    assert scored[1] == pytest.approx(3.0)
    assert scored[2] == pytest.approx(-1.0), "a negative weight has to be able to push a row DOWN"


def test_ranked():
    """Members enter the fit as ranks in [0, 1] — the same space the flat vote works in."""
    ranks = StackCombiner(_pool()).ranked()

    assert ranks.shape == (STUDIES, MEMBERS, len(TARGETS))
    assert 0.0 <= ranks.min() <= ranks.max() <= 1.0
    column = ranks[:, 0, 0]
    original = _pool().predictions[0][:, 0]
    assert np.all(np.diff(column[original.argsort()]) >= 0.0), (
        "ranking must not reorder a member; it only rescales what that member already said"
    )
    assert len(np.unique(column)) == len(np.unique(original)), "tied predictions share one midrank"


def test_design():
    """Own-finding features are one column per member; crossing widens them by every other finding."""
    ranks = StackCombiner(_pool()).ranked()

    own = StackCombiner(_pool()).design(ranks, 0)
    crossed = StackCombiner(_pool(), cross=True).design(ranks, 0)

    assert own.shape == (STUDIES, MEMBERS)
    assert crossed.shape == (STUDIES, MEMBERS * len(TARGETS))
    assert np.array_equal(own, ranks[:, :, 0]), "the own-finding case must be exactly that finding's columns"


def test_blocked():
    """The blocks partition the rows — every study reported once, by a fit that did not see it."""
    blocks = StackCombiner(_pool(), blocks=5).blocked(STUDIES)

    assert len(blocks) == 5
    assert np.array_equal(np.sort(np.concatenate(blocks)), np.arange(STUDIES)), "no row scored twice, none dropped"
    assert all(np.array_equal(block, np.sort(block)) for block in blocks), "contiguous, so no seed enters the reading"


def test_fittable():
    """A finding with one class present is not a hard fit, it is an unanswerable question."""
    assert StackCombiner.fittable(np.array([0, 1, 0], dtype=np.int64))
    assert not StackCombiner.fittable(np.ones(5, dtype=np.int64)), "all positive prices no ordering"
    assert not StackCombiner.fittable(np.zeros(5, dtype=np.int64)), "and neither does all negative"


def test_prior():
    """The prior IS the flat vote — every member equally, and only the target's findings."""
    combiner = StackCombiner(_pool())
    crossing = StackCombiner(_pool(), cross=True)

    own = combiner.prior(MEMBERS, 0)
    crossed = crossing.prior(MEMBERS * len(TARGETS), 1)

    assert own == pytest.approx(np.full(MEMBERS, 1.0 / MEMBERS)), "the vote weighs every member the same"
    assert crossed[MEMBERS : 2 * MEMBERS] == pytest.approx(np.full(MEMBERS, 1.0 / MEMBERS))
    assert crossed[:MEMBERS] == pytest.approx(np.zeros(MEMBERS)), (
        "a vote reads one finding's columns, so the other findings start at zero and must be earned"
    )


def test_fitted():
    """A tiny penalty recovers a real relation; a crushing one returns the prior it shrinks toward."""
    design = np.column_stack([np.arange(STUDIES, dtype=np.float64), np.ones(STUDIES)])
    response = (np.arange(STUDIES) >= STUDIES // 2).astype(np.int64)
    prior = np.array([0.5, 0.5])

    weak = StackCombiner.fitted(design, response, min(PENALTIES), prior)
    crushing = StackCombiner.fitted(design, response, 1e12, prior)

    assert weak.coefficients[0] > 0.0, "the first column IS the response's order, so its weight must be positive"
    assert crushing.coefficients == pytest.approx(prior, abs=1e-6), (
        "the largest penalty has to reproduce the control exactly, or the grid does not contain the vote"
    )


def test_chosen():
    """The penalty is picked from the grid, and a degenerate inner split refuses to pick at all."""
    combiner = StackCombiner(_pool())
    ranks = combiner.ranked()
    design = combiner.design(ranks, 0)
    response = (_truth()[:, 0] >= 0.5).astype(np.int64)

    assert combiner.chosen(design, response, 0) in PENALTIES
    assert combiner.chosen(design, np.ones(STUDIES, dtype=np.int64), 0) == max(PENALTIES), (
        "one class on either side of the inner split prices nothing, so the safest end of the grid stands"
    )


def test_stacked():
    """Every study gets a score, and a finding with one class present falls back to the flat vote."""
    truth = _truth()
    truth[:, 1] = 1.0
    pool = _pool(truth)

    stacked = StackCombiner(pool, blocks=3).stacked()

    assert stacked.shape == truth.shape
    assert np.isfinite(stacked).all()
    flat = pool.combined(list(range(len(pool))))
    assert np.array_equal(stacked[:, 1], flat[:, 1]), (
        "a constant label cannot be fitted, and inventing a score there would be a number about nothing"
    )
    assert not np.array_equal(stacked[:, 0], flat[:, 0]), "a finding that CAN be fitted has to actually be fitted"


def test_reading():
    """The two sides are scored over the same rows and the delta is their difference, not a third number."""
    reading = StackCombiner(_pool(), blocks=3).reading()

    assert set(reading) == {"stacked", "vote", "delta"}
    assert reading["delta"] == pytest.approx(reading["stacked"] - reading["vote"])
    assert 0.0 <= reading["vote"] <= 1.0
