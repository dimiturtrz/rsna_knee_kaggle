"""Zero an entire residual branch for a whole sample, rescaling the survivors to keep the mean.

Ours rather than `Dinov2DropPath` because importing a modelling module's private block to rebuild it with
a different rate couples this file to transformers' internal layout, and the arithmetic is four lines.

The mask is drawn PER SAMPLE and broadcast over every remaining axis: dropping per-element would be
dropout, which is a different mechanism with a different effect size. Dividing by the keep rate is what
makes this identity in expectation, which is why `eval()` needs no correction of its own -- a member
trained under it holds no extra parameter and no buffer, and serves byte-identically.

`StochasticDepth` in `stochastic_depth` is what puts these on a trunk, with a rate that ramps over depth.
"""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

NO_DROP = 0.0


class DropPath(nn.Module):
    """The policy one residual block carries: its branch's drop rate, and the draw that applies it."""

    def __init__(self, rate: float = NO_DROP) -> None:
        super().__init__()
        self.rate = rate

    @override
    def forward(self, branch: Float[Tensor, "b ..."]) -> Float[Tensor, "b ..."]:
        if not self.rate or not self.training:
            return branch
        kept = 1.0 - self.rate
        # The shape is (batch, 1, 1, ...) so one draw governs one sample's whole branch.
        shape = (branch.shape[0], *((1,) * (branch.ndim - 1)))
        mask = torch.empty(shape, dtype=branch.dtype, device=branch.device).bernoulli_(kept)
        return branch * mask / kept
