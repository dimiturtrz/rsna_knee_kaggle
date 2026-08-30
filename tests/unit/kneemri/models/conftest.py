"""The two trunk contracts the encoder table adapts, as fakes both mirror tests can build.

Shared rather than copied because both halves of one path need them: `trunk_tokens` tests the adapter
that wears the head's contract, `backbone` tests the table deciding which trunks arrive short of it. A
copy in each file is the same twelve lines twice, and the copy nobody edits is the one that stops
describing what the trunks actually emit.
"""

from collections.abc import Callable
from typing import override

import pytest
import torch
from torch import Tensor, nn

from kneemri.models.trunk_tokens import TokenOutput


class _FeatureMap(nn.Module):
    """A convolutional trunk's contract in three lines: `last_hidden_state` of shape (n, dim, h, w)."""

    def __init__(self, dim: int, side: int) -> None:
        super().__init__()
        self.dim, self.side = dim, side

    @override
    def forward(self, pixel_values: Tensor) -> TokenOutput:
        rows = torch.arange(self.side * self.side, dtype=torch.float32)
        emitted = rows.view(1, 1, self.side, self.side).expand(len(pixel_values), self.dim, self.side, self.side)
        return TokenOutput(emitted)


class _TokenSequence(nn.Module):
    """A hierarchical transformer's contract: `last_hidden_state` of shape (n, t, dim), and no prefix."""

    def __init__(self, dim: int, tokens: int) -> None:
        super().__init__()
        self.dim, self.tokens = dim, tokens

    @override
    def forward(self, pixel_values: Tensor) -> TokenOutput:
        rows = torch.arange(self.tokens, dtype=torch.float32)
        return TokenOutput(rows.view(1, self.tokens, 1).expand(len(pixel_values), self.tokens, self.dim))


@pytest.fixture
def feature_map() -> Callable[[int, int], nn.Module]:
    """A trunk that emits a map, at a width and a square grid its caller names."""
    return lambda dim, side: _FeatureMap(dim=dim, side=side)


@pytest.fixture
def token_sequence() -> Callable[[int, int], nn.Module]:
    """A trunk that emits a token sequence, at a width and a token count its caller names."""
    return lambda dim, tokens: _TokenSequence(dim=dim, tokens=tokens)
