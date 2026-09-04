import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.fault_ruler import PAIRS_PER_LABEL, FaultRuler, PairStatistics
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.scores import LabelScores

STUDIES = 60
MEMBERS = 5
SEED = 7
CONFOUND = 2.0


@pytest.fixture
def pool() -> Pool:
    """Five members of decreasing quality over an alternating truth, so every label has both classes."""
    rng = np.random.default_rng(SEED)
    truth = np.tile(np.array([[1.0] * len(TARGETS), [0.0] * len(TARGETS)]), (STUDIES // 2, 1)).astype(np.float32)
    predictions = [truth + rng.normal(0, 0.4 + index / MEMBERS, truth.shape) for index in range(MEMBERS)]
    return Pool([f"member{index}" for index in range(MEMBERS)], predictions, truth)


@pytest.fixture
def ruler(pool: Pool) -> FaultRuler:
    return FaultRuler(pool, LabelScores(), seed=SEED)


def test_rank_correlation() -> None:
    """A monotone reordering reads +1, its reverse -1, and a constant column is undefined rather than zero."""
    values = np.array([3.0, 1.0, 4.0, 1.5, 9.0])
    assert FaultRuler.rank_correlation(values, values * CONFOUND) == pytest.approx(1.0)
    assert FaultRuler.rank_correlation(values, -values) == pytest.approx(-1.0)
    assert np.isnan(FaultRuler.rank_correlation(values, np.ones(len(values))))


def test_midranks() -> None:
    """Ties share the mean of the ordinal ranks they span, which is what keeps a constant column undefined."""
    assert FaultRuler.midranks(np.array([3.0, 1.0, 2.0])) == pytest.approx([3.0, 1.0, 2.0])
    assert FaultRuler.midranks(np.array([5.0, 5.0, 9.0])) == pytest.approx([1.5, 1.5, 3.0])


def test_errors(ruler: FaultRuler) -> None:
    """One row per member, one column per drawn study pair per label, every entry a discordance indicator."""
    errors = ruler.errors()
    assert errors.shape == (MEMBERS, PAIRS_PER_LABEL * len(TARGETS))
    assert set(np.unique(errors)) <= {0.0, 0.5, 1.0}


def test_rates(ruler: FaultRuler, pool: Pool) -> None:
    """Each member's sampled error rate reproduces its own `1 - AUC`, which is what makes the sample the metric's."""
    rates = ruler.rates(ruler.errors())
    scored = np.array([1 - LabelScores().mean(member, pool.truth) for member in pool.predictions])
    assert rates.shape == (MEMBERS,)
    assert float(np.abs(rates - scored).max()) < 0.01


def test_sampled(ruler: FaultRuler) -> None:
    """Every distinct unordered pair of a five-member pool, each drawn once."""
    drawn = ruler.sampled()
    assert len(drawn) == MEMBERS * (MEMBERS - 1) // 2
    assert len(set(drawn)) == len(drawn)
    assert all(left < right for left, right in drawn)


def test_statistics(ruler: FaultRuler) -> None:
    """One row per sampled pair, with disagreement and double-fault decomposing the two error rates."""
    rows = ruler.statistics()
    assert len(rows) == MEMBERS * (MEMBERS - 1) // 2
    assert all(isinstance(row, PairStatistics) for row in rows)
    assert all(0.0 <= row.double_fault <= 1.0 and 0.0 <= row.disagreement <= 1.0 for row in rows)
    assert all(row.spread >= 0.0 for row in rows)


def test_partial(ruler: FaultRuler) -> None:
    """A statistic whose RAW correlation is negative through a shared confound reads positive once it is held fixed."""
    rng = np.random.default_rng(SEED)
    confound, shared = rng.normal(size=200), rng.normal(size=200)
    statistic, gains = confound + shared, -CONFOUND * confound + shared
    assert FaultRuler.rank_correlation(statistic, gains) < 0
    assert ruler.partial(statistic, gains, confound.reshape(-1, 1)) > 0.9


def test_oracle(ruler: FaultRuler) -> None:
    """The line that says whether the sample reproduces the score, with the gap that decides it."""
    assert "worst gap" in ruler.oracle(ruler.errors())


def test_report(ruler: FaultRuler) -> None:
    """The header names the denominator, the oracle is printed before any correlation, and a verdict is reached."""
    lines = ruler.report()
    assert f"ruler check over {MEMBERS} members" in lines[0]
    assert "oracle" in lines[1]
    assert any("VERDICT" in line for line in lines)
    assert any("INCUMBENT" in line for line in lines)
