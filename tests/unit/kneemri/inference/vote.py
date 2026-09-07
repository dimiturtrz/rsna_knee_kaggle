import numpy as np
import pytest
from numpy.typing import NDArray

from kneemri.inference.vote import (
    CHANCE,
    ColumnWeightedLogit,
    LabelQualityRank,
    LogitAverage,
    MedianLogit,
    QualityWeightedRank,
    RankAverage,
    SharpenedRank,
    Vote,
)

STUDIES = 40
LABELS = 3


NEARLY_IDENTICAL = 0.99
"""How closely a nearly-flat squash must track the flat rule, given the bend a finite k still applies."""


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


def test_midranks():
    """Ties share the average of the positions they span, which is what makes a rank a READING.

    The constant column is the case that decides it: a member emitting one value everywhere knows
    nothing, and under ordinal ranks it votes a full sweep from 0 to 1 ordered by corpus position.
    """
    assert Vote.midranks(np.array([3.0, 1.0, 2.0])).tolist() == [2.0, 0.0, 1.0]
    assert Vote.midranks(np.array([1.0, 1.0, 1.0])).tolist() == [1.0, 1.0, 1.0], (
        "a member that knows nothing votes flat"
    )
    assert Vote.midranks(np.array([5.0, 1.0, 1.0, 9.0])).tolist() == [2.0, 0.5, 0.5, 3.0]


def test_ranks_ties_vote_flat():
    """A tied member contributes NOTHING to an ensemble, instead of contributing the corpus ordering."""
    saturated = np.full((STUDIES, LABELS), 0.999)
    strong = _members(1)[0]
    combined = RankAverage().combine([strong, saturated])

    assert np.allclose(Vote.ranks(saturated), 0.5), "every study ties, so every study takes the midrank"
    assert np.array_equal(np.argsort(combined[:, 0]), np.argsort(Vote.ranks(strong)[:, 0])), (
        "a flat vote cannot reorder the member that does know something"
    )


def test_vote_named():
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


def test_median_logit_combine():
    """The median ignores an outlier the mean cannot, and agrees with the mean when nothing is one.

    Two members near 0.5 and one saturated at 1.0: the mean is dragged above 0.5 by the outlier, the
    median is not, because it moves only when the ordering of the middle changes.
    """
    outlier = np.full((STUDIES, LABELS), 1.0)
    near = [np.full((STUDIES, LABELS), 0.5), np.full((STUDIES, LABELS), 0.51)]
    dragged = LogitAverage().combine([*near, outlier])
    resisted = MedianLogit().combine([*near, outlier])

    assert np.isfinite(resisted).all(), "the saturated member is clipped here too"
    assert (resisted < dragged).all(), "the median resists the outlier the mean is dragged by"
    assert np.allclose(resisted, 0.51), "and it sits on the middle member rather than between the three"
    assert np.allclose(MedianLogit().combine(near * 2), LogitAverage().combine(near * 2), atol=0.01), (
        "with no outlier the two rules answer nearly the same number"
    )


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


def test_label_quality_rank_combine():
    """A member at chance ON ONE LABEL is excluded THERE and still votes everywhere else.

    That is the whole difference from `QualityWeightedRank`, which has one weight per member and so
    cannot express a member that knows fractures and not menisci — the case the per-label profiles in
    `weights_gap` say the pool actually contains.
    """
    strong, mixed = _members(2)
    per_label = np.full((2, strong.shape[1]), 0.9)
    per_label[1, 0] = CHANCE

    weighted = LabelQualityRank(per_label).combine([strong, mixed])

    assert np.allclose(weighted[:, 0], Vote.ranks(strong)[:, 0]), "a member at chance on a label gets no vote there"
    assert not np.allclose(weighted[:, 1], Vote.ranks(strong)[:, 1]), "and still votes on the labels it knows"


def test_label_quality_rank_votes_flat_on_a_label_nobody_beats_chance_on():
    """Emptiness is a PER-LABEL fact, so the fallback is per label and not a whole-rule flat vote."""
    members = _members(3)
    per_label = np.full((3, members[0].shape[1]), 0.9)
    per_label[:, 2] = 0.3

    rule = LabelQualityRank(per_label)

    assert rule.weights[:, 2] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert rule.weights[:, 0] == pytest.approx([1 / 3, 1 / 3, 1 / 3]), "equal knowledge is equal say"


def test_sharpened_rank_named():
    """The strength is IN the name, because two sharpnesses are two rules in one comparison."""
    assert SharpenedRank(8.0).named() == "SharpenedRank(8)"


def test_sharpened_rank_squashed():
    """A monotone bend around the median: the tails move toward 0 and 1, the median does not move."""
    squashed = SharpenedRank(8.0).squashed(np.array([[0.0], [0.5], [1.0]]))

    assert squashed[1, 0] == pytest.approx(0.5), "the median is the pivot"
    assert squashed[0, 0] < 0.5 < squashed[2, 0]
    assert squashed[2, 0] - squashed[1, 0] < 0.5 - 0.0, "the bend compresses the range it is applied to"


def test_sharpened_rank_combine():
    """Nearly flat sharpening reproduces the flat rule's ORDER, which is what makes k a deviation from it.

    Not its values — a logistic is affine only in the limit — so the check is on the ordering the metric
    reads, through the rank correlation rather than through `argsort`: a flat rank-average of three
    members ties by construction and `argsort` settles a tie by position, which would make this test read
    array order rather than agreement. It is a floor and not an equality because a logistic is affine
    only in the limit: the bend is still present at k = 1e-3 and moves the ordering wherever two studies
    sit within one squash-step of each other, so the correlation lands near 1 without reaching it.
    """
    members = _members(3)

    gentle = SharpenedRank(1e-3).combine(members)
    flat = RankAverage().combine(members)

    assert RankAverage.agreement(gentle, flat) > NEARLY_IDENTICAL
    assert np.array_equal(SharpenedRank(8.0).combine(members[:1]), members[0]), "a lone member is untouched"


def test_column_weighted_logit_combine():
    """A uniform table reproduces `LogitAverage` to the last representable digit, and a column's own
    weights move only that column.

    To floating tolerance and not bit-for-bit: `mean(0)` and a sum of products by 1/M are the same
    number in real arithmetic and different roundings in this one, so the equality is asserted at 1e-15
    absolute rather than by `array_equal`. Nothing shipped depends on the tighter claim — every kernel
    on the board constructs `LogitAverage` itself and is untouched by this class existing — but the
    weighted table is measured as a deviation from the flat number, so the two must agree far below the
    0.0028 the deviation is worth.
    """
    members = _members(3)
    uniform = np.full((len(members), LABELS), 1 / len(members))

    np.testing.assert_allclose(
        ColumnWeightedLogit(uniform).combine(members), LogitAverage().combine(members), rtol=0, atol=1e-15
    )

    tilted = uniform.copy()
    tilted[:, 0] = [1.0, 0.0, 0.0]
    moved = ColumnWeightedLogit(tilted).combine(members)

    np.testing.assert_allclose(moved[:, 0], members[0][:, 0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(moved[:, 1:], LogitAverage().combine(members)[:, 1:], rtol=0, atol=1e-15)


def test_column_weighted_logit_refuses_a_table_that_is_not_a_mixture():
    """A negative weight and an empty column are both fits pretending to be votes, and both refuse."""
    with pytest.raises(ValueError, match="non-negative"):
        ColumnWeightedLogit(np.array([[1.5, 0.5], [-0.5, 0.5]]))

    with pytest.raises(ValueError, match="no vote"):
        ColumnWeightedLogit(np.array([[0.0, 0.5], [0.0, 0.5]]))
