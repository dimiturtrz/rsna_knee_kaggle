"""Unit cover for the transductive-adaptation probe.

The equivalence classes here are about the MECHANICS — a tail is the size it claims, a refit produces a
different head, an area is one number per finding — not about whether adaptation pays. That question is
a measurement over cached features and belongs to the module's own run, never to a unit test.
"""

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.feature_probe import ProbeSplit
from kneemri.evaluation.scores import LabelScores
from kneemri.evaluation.transduction import AdaptationResult, TestAdaptation

PENALTY = 10.0
STUDIES = 60
DIMENSION = 8


@pytest.fixture(name="corpus")
def fixture_corpus() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A separable synthetic corpus: features, truth, holdout positions, training positions."""
    generator = np.random.default_rng(0)
    truth = generator.integers(0, 2, size=(STUDIES, len(TARGETS)))
    features = generator.normal(size=(STUDIES, DIMENSION)) + truth[:, :1]
    positions = np.arange(STUDIES)
    return features, truth, positions[:20], positions[20:]


PREVALENCE = np.full(len(TARGETS), 0.5)
"""Half of every finding, so a prevalence cap can never bind in a test written for the fixed scheme."""


def test_pseudo_labels() -> None:
    """Each finding's tails are the requested size, and only the tails count as confident."""
    adaptation = TestAdaptation(PENALTY, share=0.25)
    predictions = np.random.default_rng(1).normal(size=(40, len(TARGETS)))
    labels, weights = adaptation.pseudo_labels(predictions, PREVALENCE)
    assert labels.shape == predictions.shape
    assert labels.sum(axis=0).tolist() == [10] * len(TARGETS)
    assert weights.sum(axis=0).tolist() == [20.0] * len(TARGETS)


def test_pseudo_labels_tiny_test_set() -> None:
    """A test set too small for the share still yields one study a tail, never zero."""
    adaptation = TestAdaptation(PENALTY, share=0.05)
    predictions = np.random.default_rng(2).normal(size=(3, len(TARGETS)))
    labels, weights = adaptation.pseudo_labels(predictions, PREVALENCE)
    assert labels.sum(axis=0).tolist() == [1] * len(TARGETS)
    assert weights.sum(axis=0).tolist() == [2.0] * len(TARGETS)


def test_adapted(corpus: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    """The refit reports both reads over the holdout it was given, and names the trunk it was asked for."""
    features, truth, holdout, training = corpus
    result = TestAdaptation(PENALTY, share=0.25).adapted("synthetic", ProbeSplit(features, truth, training, holdout))
    assert result.trunk == "synthetic"
    assert result.studies == len(holdout)
    assert set(result.fitted) == set(TARGETS)
    assert set(result.adapted) == set(TARGETS)


def test_areas() -> None:
    """One area per finding, in the target order, each a probability."""
    generator = np.random.default_rng(3)
    held = np.tile(np.array([[0], [1]]), (10, len(TARGETS)))
    areas = TestAdaptation.areas(generator.normal(size=(20, len(TARGETS))), held)
    assert list(areas) == list(TARGETS)
    assert all(0.0 <= area <= 1.0 for area in areas.values())


def test_delta() -> None:
    """The delta is the mean adapted area minus the mean fitted area."""
    result = AdaptationResult(
        trunk="t",
        labelling="fixed",
        share=0.25,
        studies=9,
        fitted=dict.fromkeys(TARGETS, 0.80),
        adapted=dict.fromkeys(TARGETS, 0.75),
    )
    assert result.delta == pytest.approx(-0.05)


def test_line() -> None:
    """The printed row carries the trunk, the share and a signed delta."""
    result = AdaptationResult(
        trunk="ortholarge",
        labelling="prevalence",
        share=0.1,
        studies=909,
        fitted=dict.fromkeys(TARGETS, 0.80),
        adapted=dict.fromkeys(TARGETS, 0.81),
    )
    line = result.line()
    assert "ortholarge" in line
    assert "n=909" in line
    assert "+0.0100" in line


def test_areas_match_label_scores_with_a_one_class_column() -> None:
    """Every area equals the rank AUC per column, and a one-class column is NaN in the same place."""
    generator = np.random.default_rng(11)
    held = generator.integers(0, 2, size=(40, len(TARGETS)))
    held[:, 2] = 1
    predictions = generator.normal(size=held.shape)
    expected = {
        target: LabelScores.area_under_curve(predictions[:, column], held[:, column])
        for column, target in enumerate(TARGETS)
    }
    areas = TestAdaptation.areas(predictions, held)
    np.testing.assert_equal(areas, expected)
    assert np.isnan(areas[TARGETS[2]])
