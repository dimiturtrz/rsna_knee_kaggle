import numpy as np
import pytest

from kneemri.evaluation.scores import LabelScores
from kneemri.models.model import TARGETS


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

    Raw sigmoids effectively never tie, so single-member scores never moved. `Ensemble.combine` maps
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
    interval = scores.paired_interval(first, second, truth, resamples=120)
    observed = scores.per_label(second, truth)["ACL"] - scores.per_label(first, truth)["ACL"]
    low, high = interval["ACL"]

    assert low <= observed <= high, "the band must bracket the delta it bounds"
    assert (high - low) < 0.067, "a paired band must beat the unpaired bound it replaces"
    assert scores.paired_interval(first, second, truth, resamples=120) == interval, "seeded, so it is quotable"
