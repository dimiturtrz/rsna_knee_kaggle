import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.scores import ArmPair, LabelScores


def test_area_under_curve():
    """A perfect ranking scores 1, a reversed one 0, and a single-class label is undefined."""
    truth = np.array([0, 0, 1, 1], dtype=np.int64)
    assert LabelScores.area_under_curve(np.array([0.1, 0.2, 0.8, 0.9]), truth) == 1.0
    assert LabelScores.area_under_curve(np.array([0.9, 0.8, 0.2, 0.1]), truth) == 0.0
    undefined = LabelScores.area_under_curve(np.array([0.5, 0.6]), np.array([1, 1], dtype=np.int64))
    assert np.isnan(undefined), "one class present must be NaN, not 0.5 — that would fake chance"


def test_area_under_curve_is_invariant_to_monotone_maps():
    """The metric reads order only, so any increasing transform of the scores must not move it."""
    truth = np.array([0, 1, 0, 1], dtype=np.int64)
    predictions = np.array([0.2, 0.9, 0.4, 0.7])
    plain = LabelScores.area_under_curve(predictions, truth)
    assert LabelScores.area_under_curve(predictions * 100 + 5, truth) == plain
    assert LabelScores.area_under_curve(np.log(predictions), truth) == plain


def test_per_label():
    """One AUC per finding, against graded targets cut at their midpoint."""
    predictions = np.tile(np.array([[0.1], [0.9]]), (1, len(TARGETS)))
    targets = np.tile(np.array([[0.0], [1.0]], dtype=np.float32), (1, len(TARGETS)))
    per_label = LabelScores().per_label(predictions, targets)
    assert set(per_label) == set(TARGETS)
    assert per_label["ACL"] == 1.0


def test_mean():
    """The reported score is the unweighted mean over the twelve findings."""
    predictions = np.tile(np.array([[0.1], [0.9]]), (1, len(TARGETS)))
    targets = np.tile(np.array([[0.0], [1.0]], dtype=np.float32), (1, len(TARGETS)))
    assert LabelScores().mean(predictions, targets) == 1.0


def test_mean_of():
    """Unweighted, and an undefined label is skipped rather than dragging the score to nan."""
    assert LabelScores().mean_of({"a": 0.2, "b": 0.8}) == 0.5
    assert LabelScores().mean_of({"a": 0.2, "b": float("nan")}) == 0.2


def test_per_label_selection_drops_pairs_rather_than_calling_them_negative():
    """An unreadable pair must be DROPPED, never scored as a negative.

    Targets.load leaves a struck pair at the 0.5 default, which binarises to zero. Scoring a
    confidence-filtered table without a selection would therefore assert that every finding the reader
    could not read was ABSENT - a ruler more biased than the one it replaces.

    The fixture makes the difference visible: study 1 is UNKNOWN and the model ranks it top. Dropped,
    the model is perfect. Counted as a negative, it sits above the only positive.
    """
    column = np.array([0.9, 0.95, 0.1, 0.2], dtype=np.float64)
    graded = np.array([1.0, 0.5, 0.0, 0.5], dtype=np.float32)
    readable = np.array([True, False, True, False])

    predictions = np.tile(column[:, None], (1, len(TARGETS)))
    targets = np.tile(graded[:, None], (1, len(TARGETS)))
    keep = np.tile(readable[:, None], (1, len(TARGETS)))

    dropped = LabelScores().per_label(predictions, targets, keep)
    assert all(value == 1.0 for value in dropped.values()), "on the readable pairs the ranking is perfect"

    as_negative = LabelScores().per_label(predictions, targets)
    assert all(value < 1.0 for value in as_negative.values()), "the unknown rows counted as negatives change it"


def test_area_under_curve_gives_ties_midranks():
    """A constant predictor scores chance whatever the label order, and partial ties interpolate.

    This was `argsort().argsort() + 1` — ORDINAL ranks — so tied predictions were ordered by their
    position in the array and a tie was settled by STUDY ORDER. A constant predictor, carrying no
    information at all, scored 1.0 when the positives happened to sit late in the corpus and 0.0 when
    they sat early. Both orders are asserted here because a fix that only handles one is not a fix.

    Raw sigmoids effectively never tie, so single-member scores never moved. `RankAverage.combine` maps
    each member onto k/(n-1) and averages, which ties 20.5% of 881 studies at two members and 14.4% at
    five — and that is the number `member_curve` uses to choose what to submit.
    """
    late = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    early = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    constant = np.full(6, 0.5)

    assert LabelScores.area_under_curve(constant, late) == 0.5
    assert LabelScores.area_under_curve(constant, early) == 0.5

    # two positives ranked clear, the third tied with all three negatives: (2*3 + 0.5*3) / 9
    partial = np.array([0.5, 0.5, 0.5, 0.5, 0.8, 0.9])
    assert LabelScores.area_under_curve(partial, late) == pytest.approx(0.8333333, abs=1e-6)


def test_paired_interval():
    """Two arms scored on the same studies share their sampling error, so the paired band is tighter.

    The unpaired bound - each label's own error from its positive count - is what run_comparison
    prints, and at 200 positives it spans about 0.067. Two arms that differ only by a small
    perturbation must land far inside that, or the interval is not measuring the difference at all.

    Also pins the interval BRACKETS the observed delta, which is the property that makes it usable as
    a floor: a band that excludes the number it is meant to bound would refuse every real result.
    """
    generator = np.random.default_rng(7)
    studies = 400
    truth = np.zeros((studies, len(TARGETS)), dtype=np.float32)
    for column in range(len(TARGETS)):
        truth[generator.choice(studies, 120, replace=False), column] = 1.0
    signal = truth + generator.normal(0, 1.0, truth.shape)
    first = (signal + generator.normal(0, 0.05, truth.shape)).astype(np.float64)
    second = (signal + generator.normal(0, 0.05, truth.shape)).astype(np.float64)

    scores = LabelScores()
    interval = scores.paired_interval(ArmPair(first, second, truth, 120))
    observed = scores.per_label(second, truth)["ACL"] - scores.per_label(first, truth)["ACL"]
    low, high = interval["ACL"]

    assert low <= observed <= high, "the band must bracket the delta it bounds"
    assert (high - low) < 0.067, "a paired band must beat the unpaired bound it replaces"
    assert scores.paired_interval(ArmPair(first, second, truth, 120)) == interval, "seeded, so it is quotable"


def test_mean_interval():
    """One arm's absolute mean carries a band, and it is WIDER than the paired band on the same data.

    That ordering is the whole point of the method: the paired interval cancels the per-study difficulty
    two arms share, and an absolute mean cannot. Quoting the paired floor against an absolute number -
    a member's mean against a forum post, a holdout pushed through the board transfer - therefore uses
    the narrower ruler and overstates the claim.
    """
    generator = np.random.default_rng(7)
    studies = 400
    truth = np.zeros((studies, len(TARGETS)), dtype=np.float32)
    for column in range(len(TARGETS)):
        truth[generator.choice(studies, 120, replace=False), column] = 1.0
    signal = truth + generator.normal(0, 1.0, truth.shape)
    first = (signal + generator.normal(0, 0.05, truth.shape)).astype(np.float64)
    second = (signal + generator.normal(0, 0.05, truth.shape)).astype(np.float64)

    scores = LabelScores()
    low, high = scores.mean_interval(first, truth, resamples=120)
    paired = scores.paired_interval(ArmPair(first, second, truth, 120))["ACL"]

    assert low <= scores.mean(first, truth) <= high, "the band must bracket the mean it bounds"
    assert (high - low) > (paired[1] - paired[0]), "an unpaired band cannot be tighter than a paired one"
    assert scores.mean_interval(first, truth, resamples=120) == (low, high), "seeded, so it is quotable"


def test_paired_mean_interval():
    """The competition metric's own comparison band — and it must be TIGHTER the more the arms share.

    Two assertions, because the method's whole claim is the second one. That a band brackets its delta
    is table stakes; that pairing collapses as the two arms converge is the reason the borrowed
    arm-to-arm floor is the wrong ruler for two rosters built from mostly the same members.
    """
    generator = np.random.default_rng(19)
    studies = 400
    truth = np.zeros((studies, len(TARGETS)), dtype=np.float32)
    for column in range(len(TARGETS)):
        truth[generator.choice(studies, 120, replace=False), column] = 1.0
    signal = truth + generator.normal(0, 1.0, truth.shape)
    first = (signal + generator.normal(0, 0.5, truth.shape)).astype(np.float64)
    distant = (signal + generator.normal(0, 0.5, truth.shape)).astype(np.float64)
    near = (first + generator.normal(0, 0.02, truth.shape)).astype(np.float64)

    scores = LabelScores()
    low, high = scores.paired_mean_interval(ArmPair(first, distant, truth, 120))
    observed = scores.mean(distant, truth) - scores.mean(first, truth)
    tight = scores.paired_mean_interval(ArmPair(first, near, truth, 120))

    assert low <= observed <= high, "the band must bracket the delta it bounds"
    assert (tight[1] - tight[0]) < (high - low), "a near-identical arm must read a narrower band"
    assert scores.paired_mean_interval(ArmPair(first, distant, truth, 120)) == (low, high), "seeded"


def test_cut_for():
    """A per-column override beats the instrument's scalar, and every other column keeps the scalar."""
    scores = LabelScores(0.75, {TARGETS[0]: 0.5})
    assert scores.cut_for(TARGETS[0]) == 0.5, "the named finding takes its own cut"
    assert scores.cut_for(TARGETS[1]) == 0.75, "an unnamed finding takes the instrument's"
    assert LabelScores().cut_for(TARGETS[0]) == 0.5, "no overrides means the scalar everywhere"


def test_cut_for_reaches_the_metric():
    """The override has to reach the binarisation, or naming a cut is decoration.

    One column's truth sits between the two cuts, so it is a positive at the midpoint and a negative at
    0.75 — and the predictions rank it so that flipping it flips that label's AUC from 1 to 0.
    """
    predictions = np.tile(np.array([[0.9], [0.1]]), (1, len(TARGETS)))
    targets = np.tile(np.array([[0.68], [0.0]], dtype=np.float32), (1, len(TARGETS)))

    midpoint = LabelScores().per_label(predictions, targets)
    recut = LabelScores(0.75, {TARGETS[0]: 0.5}).per_label(predictions, targets)

    assert midpoint[TARGETS[1]] == 1.0, "at the midpoint the 0.68 row is a positive, ranked first"
    assert np.isnan(recut[TARGETS[1]]), "at 0.75 it is a negative, leaving that label single-class"
    assert recut[TARGETS[0]] == 1.0, "the overridden column still reads at the midpoint"


def test_name():
    """The instrument prints what it is, because a macro is not comparable across cuts."""
    assert LabelScores().name() == "cut0.5"
    assert LabelScores(0.75).name() == "cut0.75"
    assert LabelScores(0.75, {TARGETS[0]: 0.5}).name() == "ledger75", "a mixed cut is the project's ruler"


def test_drawn():
    """A draw takes the same rows from all three arrays and keeps the resample count."""
    first = np.arange(12, dtype=np.float64).reshape(4, 3)
    pair = ArmPair(first, first * 2, (first > 5).astype(np.float32), 7)
    drawn = pair.drawn(np.array([3, 0], dtype=np.int64))
    assert np.array_equal(drawn.first, first[[3, 0]])
    assert np.array_equal(drawn.second, first[[3, 0]] * 2)
    assert np.array_equal(drawn.targets, pair.targets[[3, 0]])
    assert drawn.resamples == 7


def test_difference():
    """Second minus first under the same measure, so a better second arm reads positive."""
    predictions = np.tile(np.array([[0.1], [0.9]]), (1, len(TARGETS)))
    targets = np.tile(np.array([[0.0], [1.0]], dtype=np.float32), (1, len(TARGETS)))
    pair = ArmPair(1 - predictions, predictions, targets)
    assert pair.difference(LabelScores().mean) == 1.0
