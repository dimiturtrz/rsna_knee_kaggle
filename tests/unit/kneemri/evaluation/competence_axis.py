"""Whether a pool spans one trade-off axis, on matrices built so the right answer is known in advance."""

import numpy as np

from core.findings import TARGETS
from kneemri.evaluation.competence_axis import AxisReading, CompetenceAxis, CompetenceTable
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.scores import LabelScores

MEMBERS = 12
STUDIES = 40
SEED = 2026
RESAMPLES = 60
"""Enough shuffles for the bar to separate the two classes below, few enough to keep the unit fast."""


def _traded(strengths: list[float]) -> np.ndarray:
    """A members x findings matrix whose ONLY structure is one trade: half the findings up, half down.

    Every member sits at its own point on one line, so the double-centred residual is exactly rank one
    and the first component must take essentially all of it.
    """
    direction = np.array([1.0 if column < len(TARGETS) // 2 else -1.0 for column in range(len(TARGETS))])
    return 0.8 + np.outer(np.array(strengths), direction) * 0.05


def _pool(inverted: int) -> Pool:
    """A pool where `inverted` members answer every odd finding backwards — a shape a table can read."""
    generator = np.random.default_rng(SEED)
    truth = (generator.random((STUDIES, len(TARGETS))) > 0.5).astype(np.float32)
    predictions = []
    for index in range(MEMBERS):
        member = truth + generator.normal(0.0, 0.6, truth.shape)
        if index < inverted:
            member[:, 1::2] = -member[:, 1::2]
        predictions.append(member)
    return Pool([f"m{index}" for index in range(MEMBERS)], predictions, truth)


def test_scored():
    """One AUC per member per finding, and a finding no member can be scored on is dropped, not imputed."""
    pool = _pool(inverted=0)
    truth = pool.truth.copy()
    truth[:, 0] = 1.0
    blind = Pool(pool.names, pool.predictions, truth)

    table = CompetenceTable.scored(blind, LabelScores())

    assert table.matrix.shape == (MEMBERS, len(TARGETS) - 1), "the single-class finding leaves the denominator"
    assert TARGETS[0] not in table.findings
    assert np.isfinite(table.matrix).all(), "and nothing undefined survives into the arithmetic"
    assert table.names == pool.names, "the rows stay named in the pool's order"


def test_residual():
    """A uniform row lift is member quality and a uniform column lift is difficulty; both leave nothing."""
    matrix = np.full((3, len(TARGETS)), 0.80)
    matrix[0, :] += 0.06
    matrix[:, 0] += 0.04

    residual = CompetenceTable(["a", "b", "c"], list(TARGETS), matrix).residual()

    assert np.allclose(residual, 0.0), "level and difficulty are exactly what the residual removes"


def test_read():
    """Two equivalence classes: a pool built ON one axis resolves, and a pool of noise does not.

    The second class is the one that matters. A first component always exists and always takes the
    largest share, so a reading that cannot fail to resolve says nothing — the null bar is what makes
    the verdict a measurement, and the noise pool is where it is checked.
    """
    axis = CompetenceAxis(seed=SEED)
    names = [f"m{index}" for index in range(MEMBERS)]
    traded = _traded([float(index) - MEMBERS / 2 for index in range(MEMBERS)])

    reading = axis.read(CompetenceTable(names, list(TARGETS), traded), resamples=RESAMPLES)

    assert reading.resolved, "a rank-one trade is the case this instrument exists to detect"
    assert reading.share > reading.null
    assert set(reading.loadings) == set(TARGETS)
    assert len(reading.extremes) == MEMBERS, "every member is placed, not only the ends"

    noise = 0.8 + np.random.default_rng(SEED).normal(0.0, 0.05, (MEMBERS, len(TARGETS)))

    assert not axis.read(CompetenceTable(names, list(TARGETS), noise), resamples=RESAMPLES).resolved, (
        "unlinked per-finding differences may never be reported as a shared axis"
    )


def test_strongest():
    """Ranks on the RESIDUAL, so a member mediocre overall but strong on the subset sorts to the top.

    That is the whole point of the method and the one behaviour a mean-based ranking does not have: the
    matrix below gives one member a uniformly high row and another a high row ONLY on the wanted finding,
    and the second must win.
    """
    names = ["broadly_good", "specialist", "flat"]
    matrix = np.full((len(names), len(TARGETS)), 0.80)
    matrix[0, :] = 0.86
    matrix[1, 0] = 0.86
    table = CompetenceTable(names, list(TARGETS), matrix)

    ranked = table.strongest((TARGETS[0],))

    assert ranked[0][0] == "specialist", "a uniform lift is member quality and is removed before ranking"
    assert ranked[0][1] > 0.0 > ranked[-1][1], "and the ranking is signed, so a member can be BELOW its own level"
    assert len(ranked) == len(names), "every member is placed, not only the shortlist the CLI prints"

    try:
        table.strongest(("Not A Finding",))
    except ValueError as refused:
        assert "Not A Finding" in str(refused), "a subset this pool cannot score is named, never silently dropped"
    else:
        raise AssertionError("a finding outside the pool must refuse")


def test_resolved():
    """The verdict is the share against the bar, and a share BELOW its own null is never a finding."""
    fields = {"findings": list(TARGETS), "quality": 0.5, "loadings": {}, "spread": {}, "extremes": [], "maturity": 0.0}

    assert AxisReading(share=0.6, null=0.3, **fields).resolved
    assert not AxisReading(share=0.3, null=0.3, **fields).resolved, "ties go to no-axis, not to the claim"


def test_lines():
    """The share never prints without its bar, and a loading never without the spread that reads it."""
    reading = AxisReading(
        findings=list(TARGETS),
        share=0.46,
        null=0.28,
        quality=0.82,
        loadings=dict.fromkeys(TARGETS, 0.2),
        spread=dict.fromkeys(TARGETS, 0.014),
        maturity=0.9,
        extremes=[(f"m{index}", 0.01 * index) for index in range(MEMBERS)],
    )

    text = "\n".join(reading.lines())

    assert "46.0%" in text and "28.0%" in text, "the bar travels with the number it judges"
    assert "82.0%" in text, "and the quality share it was separated from"
    assert "+0.200 (0.0140)" in text, "a loading is unreadable without its finding's own spread"
    assert "by structure scale" in text
    assert "along the axis" in text, "the ends never print without the distribution that reads them"
    assert "r = +0.90" in text and "CONFOUNDED" in text, "an axis that tracks member quality says so"


def test_balance():
    """A spanned trade and a one-sided tail both have two ends; only the displacement share tells them apart."""
    fields = {
        "findings": list(TARGETS),
        "share": 0.46,
        "null": 0.28,
        "quality": 0.82,
        "loadings": {},
        "spread": {},
        "maturity": 0.0,
    }
    spanned = AxisReading(extremes=[("low", -0.10), ("mid", 0.0), ("high", 0.10)], **fields)
    tailed = AxisReading(extremes=[("tail", -0.90), ("bulk", 0.01), ("bulk2", 0.02)], **fields)

    assert "both ends populated" in spanned.balance()
    assert "LOPSIDED" in tailed.balance(), "one member carrying the displacement is not a trade the pool spans"
    assert "97%" in tailed.balance(), "the share is of displacement, so a bulk of many stays a bulk"
    assert "1 of 3 members below zero" in spanned.balance(), "the side's count prints against the pool's"
