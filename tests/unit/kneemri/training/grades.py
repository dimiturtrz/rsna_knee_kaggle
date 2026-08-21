"""The grade policies, over the five levels a reader emits and the four boundaries between them."""

import pytest
import torch

from kneemri.training.grades import GRADE_LADDER, Grades, OrdinalGrades, Predicted

LEVELS = (0.08, 0.28, 0.68, 0.82, 0.94)
"""The five values the report reader emits, which the ladder restates as four crossings."""


def _graded(level: float) -> Predicted:
    """One cell carrying one reading, so a spread can be read off as a single staircase."""
    scores = torch.full((1, 1), level)
    return Predicted(torch.zeros(1, 1), scores, torch.ones_like(scores))


def test_grades_spread():
    """The null policy hands the batch back untouched — every member ever shipped trains this way."""
    batch = _graded(0.68)
    assert Grades().spread(batch) is batch


def test_grades_settings():
    """The null policy declares the ladder OFF rather than omitting it, as the other switches do."""
    assert Grades().settings() == {GRADE_LADDER: False}


def test_boundaries():
    """One boundary between each adjacent pair of levels, in the reader's own value space."""
    assert OrdinalGrades().boundaries() == pytest.approx([0.18, 0.48, 0.75, 0.88])


def test_thresholds():
    """Ordered by construction, because the agreement rates they are derived from are monotone."""
    thresholds = OrdinalGrades().thresholds()
    assert len(thresholds) == len(LEVELS) - 1
    assert thresholds == sorted(thresholds)


def test_ordinal_grades_spread():
    """Each level crosses every boundary below it and none above — the whole point of the ladder.

    Walked over all five levels rather than spot-checked: the boundary cases (the bottom level crossing
    nothing, the top crossing everything) are where an off-by-one in the midpoints would show.
    """
    crossings = [OrdinalGrades().spread(_graded(level)).scores.flatten().tolist() for level in LEVELS]
    assert crossings == [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]


def test_ordinal_grades_spread_offsets_the_logit_and_repeats_the_weight():
    """One logit becomes four shifted copies; the confidence is the cell's own, not divided among them.

    How much a reading is trusted does not depend on which boundary is being asked about, and dividing
    it would make a ladder run train at a quarter of the plain recipe's confidence.
    """
    spread = OrdinalGrades().spread(Predicted(torch.zeros(1, 1), torch.full((1, 1), 0.94), torch.full((1, 1), 0.95)))
    offsets = [-threshold for threshold in OrdinalGrades().thresholds()]
    assert spread.logits.flatten().tolist() == pytest.approx(offsets, rel=1e-6)
    assert spread.weights.flatten().tolist() == pytest.approx([0.95] * (len(LEVELS) - 1))


def test_ordinal_grades_settings():
    """A ladder run says so in the vocabulary the pre-flight landed check reads."""
    assert OrdinalGrades().settings() == {GRADE_LADDER: True}
