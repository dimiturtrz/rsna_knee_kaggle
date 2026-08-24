import numpy as np
import pytest
from numpy.typing import NDArray

from kneemri.inference.vote import CHANCE, LogitAverage, QualityWeightedRank, RankAverage, Vote

STUDIES = 40
LABELS = 3


def _members(count: int, seed: int = 2026) -> list[NDArray[np.float64]]:
    generator = np.random.default_rng(seed)
    return [generator.random((STUDIES, LABELS)) for _ in range(count)]


def _historic(per_member: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """The rank-average EXACTLY as `Ensemble` spelled it inline before the vote became a family.

    Kept here rather than deleted, because "the shipped rule did not move" is the only claim that makes
    this refactor safe, and a claim about the old code needs the old code to check against.
    """
    if len(per_member) == 1:
        return per_member[0]
    stacked = np.stack(per_member)
    ranks = np.empty_like(stacked)
    studies = stacked.shape[1]
    for index in range(stacked.shape[0]):
        for label in range(stacked.shape[2]):
            ranks[index, :, label] = stacked[index, :, label].argsort().argsort() / max(studies - 1, 1)
    return ranks.mean(0)


def test_rank_average_combine():
    """The flat rule reproduces the pre-refactor arithmetic exactly, on every arity that ships.

    EXACTLY, not approximately: this rule produced every number in the ledger and the 0.919 on the board,
    so a refactor that moved it by an ulp would silently make those numbers unreproducible.
    """
    for count in (1, 2, 4):
        members = _members(count)
        combined = RankAverage().combine(members)
        assert np.array_equal(combined, _historic(members)), f"{count} member(s)"

    lone = _members(1)
    assert np.array_equal(RankAverage().combine(lone), lone[0]), "one member is returned untouched, not ranked"


def test_ranks():
    """Ranks run per label and land in [0, 1], because the metric ranks per label.

    One ordering across labels would mix twelve unrelated sequences into one — the failure the per-label
    rule exists to prevent, so the columns must be independently ordered.
    """
    member = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
    ranked = Vote.ranks(member)

    assert ranked[:, 0].tolist() == [1.0, 0.0, 0.5]
    assert ranked[:, 1].tolist() == [0.0, 1.0, 0.5], "the second label orders on its own values"


def test_named():
    """A rule names itself, so a reported number can carry the rule that produced it."""
    assert RankAverage().named() == "RankAverage"
    assert LogitAverage().named() == "LogitAverage"


def test_quality_weighted_rank_combine():
    """A weight of zero on a member is that member excluded, which is what makes the weights readable."""
    strong, noise = _members(2)
    weighted = QualityWeightedRank([0.9, CHANCE]).combine([strong, noise])

    assert np.allclose(weighted, Vote.ranks(strong)), "a member at chance gets no vote"


def test_quality_weighted_rank_votes_flat_when_nothing_beats_chance():
    """No basis for preferring a member is not a reason to divide by zero, nor to elect one at random."""
    members = _members(3)
    rule = QualityWeightedRank([0.4, 0.5, 0.3])

    assert rule.weights == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(rule.combine(members), RankAverage().combine(members))


def test_logit_average_combine():
    """Averaging logits keeps the MARGIN a rank throws away, and a saturated member must not vote alone.

    Without the clip a member emitting exactly 1.0 votes +inf and carries the ensemble by itself — which
    bf16 training and fp16 serving both make reachable.
    """
    confident = np.full((STUDIES, LABELS), 1.0)
    unsure = np.full((STUDIES, LABELS), 0.5)
    combined = LogitAverage().combine([confident, unsure])

    assert np.isfinite(combined).all(), "a saturated member is clipped, not infinite"
    assert (combined > 0.5).all(), "and it still pulls the ensemble its way"


def test_agreement():
    """A member agrees perfectly with itself and near-zero with an independent draw."""
    first, second = _members(2)
    assert RankAverage.agreement(first, first) == 1.0
    assert abs(RankAverage.agreement(first, second)) < 0.4


def test_agreement_is_blind_to_a_monotone_rescale():
    """Ranks, not logits — so a recalibrated member is the SAME member to this ruler.

    The property that makes agreement a distance between ORDERINGS rather than between calibrations,
    and the reason it lives beside `ranks` rather than beside the report that first printed it.
    """
    member = _members(1)[0]
    assert RankAverage.agreement(member, member * 7.0 - 2.0) == 1.0
