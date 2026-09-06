"""Cross-step margin: the equality oracle first, then the widening the memory is for."""

import torch

from kneemri.training.grades import GRADE_LADDER, Predicted
from kneemri.training.label_focus import FOCUS, UNFOCUSED
from kneemri.training.objective import (
    CONTRAST_TERM,
    COUNT_NORMALISED,
    DISSENT_TERM,
    DISTILL_TERM,
    FEATURE_TERM,
    MARGIN_TERM,
    PAIR_POOL,
    PER_LABEL,
    RANK_TERM,
    MarginObjective,
)
from kneemri.training.remembered_margin import RememberedMargin
from kneemri.training.side_ranking import SIDE_TERM


def _batch(order: list[float]) -> Predicted:
    """One label's worth of logits laid out in `order`, with binary targets and full confidence."""
    logits = torch.tensor(order).reshape(-1, 1)
    scores = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
    return Predicted(logits, scores, torch.ones_like(scores))


def test_settings():
    """The pool is read off the memory, so a run's `--pair-memory` is checked against what would drop."""
    assert RememberedMargin(0.3, capacity=4).settings() == {
        RANK_TERM: 0.0,
        DISTILL_TERM: 0.0,
        FEATURE_TERM: 0.0,
        DISSENT_TERM: 0.0,
        CONTRAST_TERM: 0.0,
        MARGIN_TERM: 0.3,
        PAIR_POOL: 4,
        PER_LABEL: False,
        COUNT_NORMALISED: False,
        GRADE_LADDER: False,
        SIDE_TERM: 0.0,
        FOCUS: "",
        UNFOCUSED: 0.0,
    }


def test_auc_margin():
    """The empty memory is the equality oracle, and a remembered step must then change the loss.

    The first assertion is the one that matters: with nothing remembered this MUST be the old objective,
    or every comparison against the measured margin baseline compares two things at once. The second
    says the memory is actually consulted rather than written and ignored — a pool that never reaches
    the loss would pass the oracle perfectly while doing nothing.
    """
    batch = _batch([0.4, -1.0, 2.0, 0.5, 3.0, -2.0])
    assert torch.equal(RememberedMargin().auc_margin(batch), MarginObjective().auc_margin(batch))

    objective = RememberedMargin()
    objective.auc_margin(_batch([3.0, 3.0, 3.0, -3.0, -3.0, -3.0]))
    assert not torch.equal(objective.auc_margin(batch), MarginObjective().auc_margin(batch))


def test_widened():
    """The live batch comes first and the pool follows, over the three fields the margin reads."""
    live = _batch([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
    pooled = _batch([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    widened = RememberedMargin().widened(live, pooled)
    assert widened.logits.shape[0] == live.logits.shape[0] + pooled.logits.shape[0]
    assert torch.equal(widened.logits[: live.logits.shape[0]], live.logits)
    assert torch.equal(widened.scores[live.scores.shape[0] :], pooled.scores)
    assert widened.teacher is None, "the margin never reads a teacher, so carrying one would be a lie"
