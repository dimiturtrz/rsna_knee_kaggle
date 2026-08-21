"""Attention pooling over a slot's slices that can first see ORDER along depth."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from kneemri.models.slice_attention import GATE, SliceAttention

_SPAN = 3
"""Slices a depth mixer sees at once: a slice and its two neighbours.

The smallest span that can express "this continues into the next slice" at all, which is the claim
being tested. Wider spans are a knob on top of a mechanism that has not yet read positive.
"""


class DepthAttention(nn.Module):
    """The same attention pooling, but each slice may first see its NEIGHBOURS along depth.

    THE AXIS NO ARM IN THIS PROJECT HAS EVER VARIED. The data is 3D, and every aggregation this model
    has run over the slice axis is permutation-invariant: `over_depth` takes a maximum within depth
    regions and then a mean ACROSS them (symmetric under swapping the regions), and `SliceAttention`
    scores each slice from that slice alone. Three encoders, whole-stack reading, two slice poolings and
    a resolution sweep all sat on top of one of those two operators. So the model has never been able to
    represent that a finding EXTENDS ACROSS CONSECUTIVE SLICES — which is most of what makes a tear read
    as a tear rather than as one bright slice — and `attend`'s +0.0002 over `max` did not test order; it
    swapped one order-blind operator for another.

    The mixer is a depthwise convolution of span `_SPAN` over the slice axis: the cheapest operator that
    can express contiguity at all, at a few thousand parameters, kept to `SliceAttention`'s scale because
    this task is supervision-limited (58 of 4,407 studies carry an image-read label) and a transformer
    over twenty tokens would be answering a capacity question instead of this one.

    ABSOLUTE DEPTH IS DELIBERATELY NOT SUPPLIED. A positional embedding would say "slice 5 of 12" — a
    different claim, that WHERE in the band a finding sits matters — and mixing the two into one module
    would leave a positive result unattributable to either. Contiguity first; position is its own variant
    if this one reads.

    IT STARTS AS EXACTLY `attend`. The convolution is initialised to the identity tap — centre one,
    neighbours zero — so at step zero this module computes what the incumbent computes, and every
    difference in the arm is something training chose to add. The alternative, a randomly initialised
    mixer, starts by smearing the features it was given and would charge that cost to the mechanism.
    """

    def __init__(self, dim: int, gate: int = GATE, span: int = _SPAN) -> None:
        super().__init__()
        # No bias: the mixer is a pure tap over depth. A per-channel offset here would be added straight
        # into the features the gate then LayerNorms, so it is redundant with the gate's own affine — and
        # it is one more tensor that has to be zeroed for step zero to equal `attend`.
        mix = nn.Conv1d(dim, dim, span, padding=span // 2, groups=dim, bias=False)
        nn.init.zeros_(mix.weight)
        with torch.no_grad():
            mix.weight[:, :, span // 2] = 1.0
        self.mix = mix
        self.pool = SliceAttention(dim, gate)

    @override
    def forward(self, x: Float[Tensor, "b s g d"]) -> Float[Tensor, "b s d"]:
        """Mix each slice with its neighbours along depth, then pool the depth axis away.

        The slots are folded into the batch because the convolution is over DEPTH only: a slot is a
        different acquisition of the same knee, and letting a kernel run across the slot boundary would
        mix a sagittal slice into an axial one purely because of how they are laid out in the tensor.
        """
        batch, slots, group, dim = x.shape
        depth_first = x.reshape(batch * slots, group, dim).transpose(1, 2)
        mixed = self.mix(depth_first).transpose(1, 2).reshape(batch, slots, group, dim)
        return self.pool(mixed)
