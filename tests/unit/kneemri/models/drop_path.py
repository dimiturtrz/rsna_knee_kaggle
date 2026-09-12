"""A dropped branch takes a WHOLE sample, is identity in eval, and is identity in expectation in train."""

import pytest
import torch

from kneemri.models.drop_path import NO_DROP, DropPath

RATE = 0.5
WIDTH = 8
BATCH = 64
SEED = 2026
TOLERANCE = 0.2


def test_forward() -> None:
    """Identity in eval, and identity IN EXPECTATION in train -- that is what makes serving unaffected.

    The row check is the line between stochastic depth and dropout: a whole sample goes or stays, so
    every row is all-zero or all-(1/kept), never a mixture. A per-element mask would pass a mean test
    and be a different mechanism with a different effect size.
    """
    torch.manual_seed(SEED)
    branch = torch.ones(BATCH, WIDTH)
    path = DropPath(RATE)

    path.eval()
    assert torch.equal(path.forward(branch), branch)

    path.train()
    dropped = path.forward(branch)
    assert dropped.unique(dim=1).shape[1] == 1
    assert dropped.mean().item() == pytest.approx(1.0, abs=TOLERANCE)


def test_forward_is_identity_at_zero() -> None:
    """The null policy: an unflagged run must be byte-identical to one with no `DropPath` at all."""
    branch = torch.randn(BATCH, WIDTH)
    path = DropPath(NO_DROP)
    path.train()
    assert torch.equal(path.forward(branch), branch)
