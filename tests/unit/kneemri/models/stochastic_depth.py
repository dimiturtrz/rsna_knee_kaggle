"""The walk ramps with depth, leaves a zero rate alone, and refuses a trunk it could not reach."""

from typing import cast

import pytest
from torch import nn

from kneemri.models.drop_path import NO_DROP, DropPath
from kneemri.models.stochastic_depth import StochasticDepth

RATE = 0.5
BLOCKS = 4
WIDTH = 8


class _Block(nn.Module):
    """The shape a residual block presents to the walk: something with a `drop_path` attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.drop_path = nn.Identity()


def _rate(block: nn.Module) -> float:
    # `nn.Module.__getattr__` answers `Module | Tensor`, so a submodule read is untyped at every use
    # site. Narrowing it ONCE here is the alternative to a suppression that says nothing.
    return cast("DropPath", block.drop_path).rate


def _trunk(blocks: int = BLOCKS) -> nn.Sequential:
    return nn.Sequential(*[_Block() for _ in range(blocks)])


def test_stochastic_depth_applied() -> None:
    """The ramp is the mechanism: the first block drops nothing and the last carries the full rate."""
    trunk = _trunk()
    assert StochasticDepth(RATE).applied(trunk) == BLOCKS
    rates = [_rate(block) for block in trunk]
    assert rates[0] == pytest.approx(NO_DROP)
    assert rates[-1] == pytest.approx(RATE)
    assert rates == sorted(rates)


def test_stochastic_depth_applied_reaches_nothing_at_zero() -> None:
    """A rate of zero leaves the trunk untouched, identities included."""
    trunk = _trunk()
    assert StochasticDepth(NO_DROP).applied(trunk) == 0
    assert all(isinstance(block.drop_path, nn.Identity) for block in trunk)


def test_stochastic_depth_applied_refuses_a_trunk_it_cannot_reach() -> None:
    """THE PLUMBING WITNESS. A rate that lands on no block is an ordinary run under this arm's name.

    The checkpoint, the score and the ledger row would all be identical, so nothing downstream could
    tell the two apart -- which is why this raises where a warning would be survivable.
    """
    with pytest.raises(ValueError, match="no block"):
        StochasticDepth(RATE).applied(nn.Sequential(nn.Linear(WIDTH, WIDTH)))
