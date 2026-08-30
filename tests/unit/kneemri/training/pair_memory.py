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


def test_remember_keeps_the_embedding_detached_for_a_contrastive_pool():
    """The context is the field a widened contrastive term draws its keys from, and it is all-or-nothing.

    Detached for `test_remember`'s reason, and dropped entirely when any remembered step lacked one — a
    pool concatenated from only the steps that carried a context would align those rows against the wrong
    studies, and the `None` it falls back to is what `supcon` raises on.
    """
    memory = PairMemory(capacity=2)
    context = torch.ones(2, 1, 3, requires_grad=True)
    memory.remember(Predicted(torch.ones(2, 1), torch.ones(2, 1), torch.ones(2, 1), embedding=context))
    pooled = memory.pooled()

    assert pooled is not None
    assert pooled.embedding is not None
    assert pooled.embedding.shape == (2, 1, 3)
    assert not pooled.embedding.requires_grad

    memory.remember(Predicted(torch.ones(2, 1), torch.ones(2, 1), torch.ones(2, 1)))
    half = memory.pooled()

    assert half is not None
    assert half.embedding is None, "half a pool is worse than none — the term must raise, not narrow"
