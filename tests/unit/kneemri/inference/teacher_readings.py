"""The producer's testable half is the ORDERING — everything else is a card, a corpus and an encoder."""

from pathlib import Path

import numpy as np
import pytest
import torch

from core.data.corpus.folds import ReportFolds
from core.paths import DataRoot
from kneemri.inference.teacher_readings import TeacherReadings


def test_teach(tmp_path: Path) -> None:
    """An empty member list is refused BEFORE the thirty-six minutes, not discovered after them.

    The rest of `teach` is a card, a corpus and an encoder, so this is the one clause a unit can reach —
    and it is the clause worth reaching: a mistyped checkpoint glob expands to nothing in a shell, and
    without the guard the pass would run to completion and write a teacher file ordering no studies.
    """
    readings = TeacherReadings(DataRoot(tmp_path), torch.device("cpu"))

    with pytest.raises(ValueError, match="no checkpoints to teach from"):
        readings.teach([], ["s0"], ReportFolds(5, 2026), 0)


def test_ordered() -> None:
    member = np.array([[0.9], [0.1], [0.5]])
    assert TeacherReadings.ordered([member])[:, 0].tolist() == [1.0, 0.0, 0.5]


def test_ordered_is_invariant_to_a_monotone_rewrite_of_a_member() -> None:
    """The claim the module's docstring makes about calibration, as an executable one."""
    member = np.array([[0.9], [0.1], [0.5]])
    assert TeacherReadings.ordered([member]).tolist() == TeacherReadings.ordered([member**3]).tolist()


def test_ordered_averages_the_members_ranks() -> None:
    """A member the others outvote moves the ordering without deciding it."""
    agreeing = np.array([[0.9], [0.1], [0.5]])
    dissenting = np.array([[0.1], [0.9], [0.5]])
    combined = TeacherReadings.ordered([agreeing, agreeing, dissenting])[:, 0]
    assert combined.tolist() == [1.0, 0.0, 0.5], "two votes for the first study beat one against"


def test_ordered_spans_the_unit_interval_for_every_label() -> None:
    scores = np.random.default_rng(0).random((7, 4))
    ordered = TeacherReadings.ordered([scores])
    assert ordered.min(axis=0).tolist() == [0.0] * 4
    assert ordered.max(axis=0).tolist() == [1.0] * 4
