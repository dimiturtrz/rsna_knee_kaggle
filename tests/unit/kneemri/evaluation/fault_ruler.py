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
    assert lines == [
        "ruler check over 5 members, 10 sampled pairs, 48000 discordance trials each, voting RankAverage",
        "  oracle: sampled error rate against 1 - AUC over 5 members, "
        "worst gap 0.0023 (sampling error alone is ~0.0023)",
        "  each statistic signed so that HIGHER predicts MORE gain. PARTIAL is the reading — it holds the",
        "  pair's quality fixed; RAW is the same number confounded by the weak-and-distant shape of the pool:",
        "    disagreement rate: partial +0.1515   raw +0.8061",
        "    agreement (INCUMBENT, MemberDistance): partial +0.1152   raw +0.6485",
        "    disagreement-error ratio (DER): partial -0.0061   raw -0.8545",
        "    double-fault rate (negated): partial -0.0788   raw -0.8788",
        "  measured gain over the pairs: median +0.0244, best +0.0564, worst -0.0333",
        "  DER above 1 in 10 of 10 pairs",
        "  VERDICT: NO statistic reaches 0.2 here — the best is disagreement rate at +0.1515, so pair diversity "
        "does not predict offline gain in this pool and selecting on any of these four is selecting on noise",
    ], "pinned verbatim, read off the module before its report was split into sections"


def test_report_names_a_predictive_winner(ruler: FaultRuler, monkeypatch: pytest.MonkeyPatch) -> None:
    """At or above the cut the verdict names the statistic; ties on partial fall back to the raw reading."""
    monkeypatch.setattr(FaultRuler, "partial", lambda self, values, gains, controls: 0.5)
    assert ruler.report()[-1] == "  VERDICT: disagreement rate ranks gain best at +0.5000"
