import numpy as np

from kneemri.model import TARGETS
from kneemri.scores import LabelScores


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
