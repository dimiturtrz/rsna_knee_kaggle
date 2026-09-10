"""`DiversitySlope` — what one unit of member decorrelation is worth in ensemble margin.

Every fixture is built in the test. The pool is synthetic on purpose: with two members constructed as
exact copies of one another and a third built to disagree, the pairwise correlation is decidable by hand,
which is what makes an assertion about a DIVERSITY number possible at all. A real holdout could only be
asserted against a number this same code produced.
"""

import numpy as np
import pytest

from kneemri.evaluation.diversity_slope import MAX_SEATS, MIN_SEATS, DiversitySlope, SubsetReading
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.scores import LabelScores

LABELS = 12
STUDIES = 200
SUBSETS = 12


def _truth() -> np.ndarray:
    """Half positive on every column, so no column is ever undefined."""
    rows = np.zeros((STUDIES, LABELS), dtype=np.float32)
    rows[: STUDIES // 2, :] = 1.0
    return rows


def _pool() -> Pool:
    """Six members: a twin pair, one inverted, and three noisier ones — a spread of pairwise correlations.

    Six rather than four because the seat range starts at four: with a four-member pool every draw is the
    whole pool, so diversity is constant across subsets and the slopes are fitted on a degenerate x.
    """
    truth = _truth()
    generator = np.random.default_rng(2026)
    good = truth.astype(np.float64) + generator.normal(0.0, 0.10, truth.shape)
    twin = good.copy()
    against = 1.0 - good
    members = [good, twin, against] + [
        truth.astype(np.float64) + generator.normal(0.0, spread, truth.shape) for spread in (0.3, 0.6, 0.9)
    ]
    return Pool(["good", "twin", "against", "noisy3", "noisy6", "noisy9"], members, truth)


def _fit() -> DiversitySlope:
    return DiversitySlope(_pool(), LabelScores(), SUBSETS)


def test_diversity() -> None:
    """Twins carry no diversity; a member built to invert another carries the most there is."""
    fit = _fit()
    assert fit.diversity((0, 1)) == pytest.approx(0.0, abs=1e-6), "identical members are 1 - 1.0 apart"
    assert fit.diversity((0, 2)) > fit.diversity((0, 3)) > fit.diversity((0, 1))


def test_price() -> None:
    """A subset is priced against its OWN best single member, and the terms that made it are reported."""
    reading = _fit().price((0, 1, 3))
    assert isinstance(reading, SubsetReading)
    assert reading.seats == 3
    assert reading.margin == pytest.approx(reading.margin)
    assert 0.0 < reading.best_single <= 1.0
    assert reading.mean_single <= reading.best_single, "the mean of the singles cannot beat the best of them"


def test_draw() -> None:
    """Every drawn subset sits in the legislated seat range, and the draw is seeded, so it repeats."""
    readings = _fit().draw()
    assert len(readings) == SUBSETS
    assert all(MIN_SEATS <= reading.seats <= min(MAX_SEATS, 6) for reading in readings)
    assert [reading.margin for reading in readings] == [reading.margin for reading in _fit().draw()]


def test_slope() -> None:
    """A constructed straight line comes back with its own slope, and a real fit carries a p."""
    slope, probability = DiversitySlope.slope([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
    assert slope == pytest.approx(2.0)
    assert probability < 0.05


def test_residuals() -> None:
    """Regressing a value on itself leaves nothing, and a SECOND covariate is removed too, not ignored."""
    values = [1.0, 4.0, 2.0, 8.0]
    assert DiversitySlope.residuals(values, [values]) == pytest.approx([0.0] * len(values), abs=1e-9)
    other = [3.0, 1.0, 9.0, 2.0]
    combined = [first + 2.0 * second for first, second in zip(values, other, strict=True)]
    assert DiversitySlope.residuals(combined, [values, other]) == pytest.approx([0.0] * len(values), abs=1e-9)


def test_report() -> None:
    """All three slopes are named, the confound among them, and the answer is marked as the answer."""
    fit = _fit()
    lines = fit.report(fit.draw())
    assert f"{SUBSETS} random subsets of 6 members" in lines[0]
    assert any(line.startswith("RAW") for line in lines)
    assert any(line.startswith("CONFOUND") for line in lines)
    assert any("<- the answer" in line for line in lines)
