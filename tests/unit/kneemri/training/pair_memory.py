"""What the buffer keeps, and what it hands back before anything has been kept."""

import torch

from kneemri.training.grades import Predicted
from kneemri.training.pair_memory import PairMemory


def _batch(order: list[float]) -> Predicted:
    """One label's worth of logits laid out in `order`, with binary targets and full confidence."""
    logits = torch.tensor(order).reshape(-1, 1)
    scores = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
    return Predicted(logits, scores, torch.ones_like(scores))


def test_pooled():
    """The first step of a run has no memory, and must not be a special case at the call site."""
    assert PairMemory().pooled() is None


def test_capacity():
    """The deque is the authority, so a request checked against it is checked against what would drop."""
    assert PairMemory(capacity=8).capacity() == 8


def test_remember():
    """A remembered readout is a fixed target: keeping its graph would hold the whole step alive."""
    memory = PairMemory(capacity=2)
    for value in (0.0, 1.0, 2.0):
        logits = torch.full((2, 1), value, requires_grad=True)
        memory.remember(Predicted(logits, torch.ones(2, 1), torch.ones(2, 1)))
    pooled = memory.pooled()

    assert pooled is not None
    assert pooled.logits.shape == (4, 1), "capacity 2 of two studies each, the oldest dropped"
    assert not pooled.logits.requires_grad
    assert sorted({float(value) for value in pooled.logits.flatten()}) == [1.0, 2.0]
