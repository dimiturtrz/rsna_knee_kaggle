"""Unit cover for the pseudo-label schemes.

The equivalence classes are the two ways a tail can be sized: by confidence alone, and by confidence
capped at what the finding's own prevalence allows. Whether either scheme PAYS is a measurement over
cached features and belongs to `transduction`'s own run, never to a unit test.
"""

import numpy as np

from core.findings import TARGETS
from kneemri.evaluation.labelling import FixedTail, PrevalenceTail

HELD = 40
BALANCED = np.full(len(TARGETS), 0.5)
"""Half of every finding, so a prevalence cap can never bind in a test written for the fixed scheme."""


def test_fixed_tail_assigned() -> None:
    """The fixed scheme labels exactly the requested share positive, whatever a finding's prevalence is."""
    predictions = np.random.default_rng(4).normal(size=(HELD, len(TARGETS)))
    labels, weights = FixedTail().assigned(predictions, 0.25, BALANCED)
    assert labels.sum(axis=0).tolist() == [10] * len(TARGETS)
    assert weights.sum(axis=0).tolist() == [20.0] * len(TARGETS)


def test_prevalence_tail_assigned() -> None:
    """The prevalence scheme never labels more positives than the finding's own training rate allows."""
    predictions = np.random.default_rng(5).normal(size=(HELD, len(TARGETS)))
    labels, weights = PrevalenceTail().assigned(predictions, 0.25, np.full(len(TARGETS), 0.1))
    assert labels.sum(axis=0).tolist() == [4] * len(TARGETS)
    assert weights.sum(axis=0).tolist() == [14.0] * len(TARGETS)
