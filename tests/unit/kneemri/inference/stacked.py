"""The learned combiner's contract: it fits what it is given, and its weights follow the signal."""

import numpy as np

from kneemri.inference.stacked import StackedLogit


def _pool(members: int, studies: int, findings: int) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(0)
    truth = (rng.random((studies, findings)) > 0.7).astype(np.int64)
    signal = truth.astype(np.float64) * 0.4
    return [np.clip(signal + rng.normal(0.3, 0.15, (studies, findings)), 0.01, 0.99) for _ in range(members)], truth


def test_logits() -> None:
    """A saturated member is clipped rather than voting infinity."""
    assert np.isfinite(StackedLogit.logits(np.array([[0.0, 1.0]]))).all()


def test_fitted() -> None:
    """One weight per member per label, and a member the target ignores gets less than one that carries it."""
    per_member, truth = _pool(members=3, studies=200, findings=4)
    per_member[2] = np.random.default_rng(1).random((200, 4))
    stacker = StackedLogit.fitted(per_member, truth)
    assert stacker.weights.shape == (3, 4)
    assert stacker.weights[:2].mean() > stacker.weights[2].mean()


def test_combine() -> None:
    """The combined table has one member's shape, not the stack's."""
    per_member, truth = _pool(members=3, studies=50, findings=4)
    combined = StackedLogit.fitted(per_member, truth).combine(per_member)
    assert combined.shape == (50, 4)
