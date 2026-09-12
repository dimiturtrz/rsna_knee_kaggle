"""Hand the head both slice moments at DOUBLE width, which is the shape the frozen probe measured."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn


class WideMomentPooling(nn.Module):
    """The two moments handed to the head at DOUBLE width — the frozen probe's reading, without a mix.

    WHY THIS EXISTS, AND IT IS A MEASUREMENT NOT A PREFERENCE. `moment_dense` ran `MomentPooling` at the
    project's best solo corner and read **0.8516 against a 0.8634 control, -0.0118** — the bottom band.
    Both suspects that band named were then retired by dumping the checkpoint: the spread half of the mix
    is NOT dead (`mean|spread| 0.0107` against `mean|first| 0.0135`, ratio **0.79**, max 0.144), so the
    0.26x scale of the second moment did not starve it either. What the dump DID show is the third thing
    that arm's header named as a difference from what was measured, and it is now the leading one: the
    identity initialisation on the mean half is **gone** — `mean|first - I| = 0.0122` against a
    `mean|first|` of 0.0135, and `torch.eye`'s own mean magnitude at this width is only 1/768 = 0.0013.
    So the head was not fed the mean beside its spread. It was fed a fully learned 768x768 projection of
    the pair, in place of the max pool it replaced, learned from scratch inside the same 12-epoch budget.

    That is a WIDTH COLLAPSE, not a verdict on the statistic. `feature_probe --slice-readout` ordered
    `moments > mean > max` on eight of nine trunks by +0.0116 to +0.0179, and it read the two moments
    CONCATENATED — never mixed back down. This module is that reading: the concatenation, handed
    through, with `SlicePooling.out_multiplier` widening the head that receives it.

    IT HAS NO PARAMETERS, WHICH IS THE POINT. `MomentPooling` spent budget learning a projection before
    the head could use either moment; this spends none, so the only thing the 12 epochs fit is the head
    itself — exactly as under `max`. The head's first Linear doubles in width and nothing else grows,
    and an aggregator with no tensors cannot repeat `attend_dense`'s failure (knee-7gyw) of sitting in
    no optimiser group: there is nothing to leave out.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.width = dim
        """The width IN. The head is built at `MOMENTS` times this, which is what carries the reading."""

    @override
    def forward(self, x: Float[Tensor, "b s g d"]) -> Float[Tensor, "b s e"]:
        """The mean along the slice axis concatenated with its spread, at twice the width in.

        The spread is the BIASED standard deviation for the reason `MomentPooling` gives: a slot holding
        a single slice must reduce to zero rather than to `nan`, and V-JEPA2's cache carries exactly that
        shape.
        """
        return torch.cat([x.mean(dim=2), x.std(dim=2, unbiased=False)], dim=-1)
