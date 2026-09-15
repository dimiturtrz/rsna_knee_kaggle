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
    """The policy one residual block carries: its branch's drop rate, and the draw that applies it.

    The rate lives in a non-persistent buffer so a curriculum can move it every step without a recompile:
    as a Python float, dynamo guarded on its value and recompiled per step until it fell back to eager
    (probe224_grand, 2026-09-15: 2483 img/s against its control's 3395). Non-persistent keeps the
    checkpoint key set unchanged.
    """

    _rate: Tensor

    def __init__(self, rate: float = NO_DROP) -> None:
        super().__init__()
        self.register_buffer("_rate", torch.tensor(rate), persistent=False)
        self._active = rate > 0

    @property
    def rate(self) -> float:
        return float(self._rate)

    @rate.setter
    def rate(self, value: float) -> None:
        self._rate.fill_(value)
        self._active = value > 0

    @override
    def forward(self, branch: Float[Tensor, "b ..."]) -> Float[Tensor, "b ..."]:
        if not self._active or not self.training:
            return branch
        kept = 1.0 - self._rate.to(branch.dtype)
        # The shape is (batch, 1, 1, ...) so one draw governs one sample's whole branch.
        shape = (branch.shape[0], *((1,) * (branch.ndim - 1)))
        mask = (torch.rand(shape, dtype=branch.dtype, device=branch.device) < kept).to(branch.dtype)
        return branch * mask / kept
