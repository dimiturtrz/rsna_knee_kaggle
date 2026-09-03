"""Pool a slot's slices by their mean AND their spread, where every other pooling reads only the mean."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

MOMENTS = 2
"""Mean and standard deviation — the two statistics concatenated before the mix projects them back."""


class MomentPooling(nn.Module):
    """Reduce the slice axis to its first two moments, then mix them back to one slot vector.

    THE AXIS THIS OPENS. `max`, `attend`, `depth` and `labelwise` are all readings of the FIRST moment:
    a weighted or elementwise selection among slices, learned or not. None of them can express how a
    channel VARIES along the stack, which is the shape that separates a lesion occupying a few slices
    from a structure present through the whole volume. The aggregation chapter measured `attend` at
    +0.0002 over `max` and `labelwise` at the same null, and closed on that — correctly, for the axis it
    tested. This is a different axis.

    WHAT WAS MEASURED, AND HOW THIS DIFFERS FROM IT. On frozen features, a ridge over the mean
    concatenated with the standard deviation beat a ridge over the maximum on eight of nine trunks, by
    +0.0116 to +0.0179 (`feature_probe --slice-readout`, fold-0, n=909, each trunk at its own penalty;
    see `interpretations/kneemri/2026-09-03_the-frozen-probe-re-read.md`). That probe read the
    concatenation at DOUBLE width. Here the width must survive the pooling — every `SLICE_POOLINGS`
    aggregator hands back the dimension it was given, and the head's width is derived from the encoder's
    — so the two moments are mixed back down instead. That is a real difference from what was measured
    and it is the first thing to suspect if the arm reads flat: the head may simply be seeing a learned
    projection of the mean.

    IT STARTS AS EXACTLY THE MEAN. The mix is initialised to the identity on the mean half and to zeros
    on the spread half, so an untrained module is the mean over slices and learns to deviate from it.
    `SliceAttention` is zero-initialised for the same reason, and `_readout_parameters` reads this
    module's tensors by EXCLUSION from the encoder — which matters here exactly as it did there, because
    an aggregator whose parameters sit in no optimiser group scores like the thing it was initialised to
    rather than crashing (knee-7gyw).

    PERMUTATION-INVARIANT, like the pools it is compared against. Both moments are symmetric in the
    slice axis, so this sees no more of the ORDER along the stack than `attend` does — `DepthAttention`
    is the module that owns that question, and keeping them separate is what makes either readable.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mix = nn.Linear(MOMENTS * dim, dim)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        with torch.no_grad():
            self.mix.weight[:, :dim] = torch.eye(dim)

    @override
    def forward(self, x: Float[Tensor, "b s g d"]) -> Float[Tensor, "b s d"]:
        """The mean along the slice axis beside its spread, mixed back to one vector per slot.

        The spread is the BIASED standard deviation, so a slot holding a single slice reduces to zero
        rather than to `nan`. That is not a hypothetical shape: V-JEPA2's cache carries a slice axis of
        length one because it consumes the stack jointly, and an unbiased estimator would poison a whole
        trunk's forward pass with a division by zero the loss would only reveal much later.
        """
        moments = torch.cat([x.mean(dim=2), x.std(dim=2, unbiased=False)], dim=-1)
        return self.mix(moments)
