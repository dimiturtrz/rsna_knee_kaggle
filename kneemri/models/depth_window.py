"""How a slice is PRESENTED to the encoder's three input channels — copies of itself, or its neighbours.

Its own module for the same reason `slice_pooling.py` is: `model.py` sits against the god-file threshold,
and this is a self-contained pixel-axis fact rather than a piece of the model's forward pass. It is also
the only through-plane operation that happens BEFORE a feature exists, which is what separates it from
every aggregator in `slice_pooling.py`.
"""

import torch
from jaxtyping import UInt8
from torch import Tensor


class DepthWindow:
    """The through-plane window a slice is read with, and the order the encoder's channels hold it in."""

    NEIGHBOURS = (-1, 0, 1)
    """Length is fixed at the encoder's channel count; the CENTRE entry is the slice being encoded."""

    @staticmethod
    def channels(imgs: UInt8[Tensor, "b s g p q"]) -> UInt8[Tensor, "b s g c p q"]:
        """Each slice paired with its two through-plane neighbours, as the encoder's three channels.

        WITHIN A SLOT AND NEVER ACROSS ONE. The slot axis holds different acquisitions — a sagittal
        slot's slices have nothing to do with an axial slot's — so a window that ran along a flattened
        depth would hand the trunk a discontinuity at every slot boundary and call it anatomy. The gather
        is along `g`, which is the only axis on which "the next slice" means anything.

        THE EDGES CLAMP RATHER THAN WRAP. Wrapping puts the last slice of the band beside the first,
        which is the largest through-plane jump available in the volume; clamping repeats the boundary
        slice, so the first and last slices see exactly what they see today (three copies of themselves)
        and every interior slice gains real context. The arm therefore differs from its control on
        `group - 2` of `group` slices and is identical on the other two, which is the conservative
        direction for a single-variable read.

        Order is (i-1, i, i+1) and it is deliberate that the CENTRE channel holds the slice being
        encoded. Whatever the trunk's first convolution learned about the middle channel of an RGB image
        is what it applies to the slice this feature vector is supposed to be about.
        """
        depth = imgs.shape[2]
        offsets = torch.tensor(DepthWindow.NEIGHBOURS, device=imgs.device)
        index = (torch.arange(depth, device=imgs.device).unsqueeze(1) + offsets).clamp(0, depth - 1)
        return imgs[:, :, index.reshape(-1)].unflatten(2, (depth, len(DepthWindow.NEIGHBOURS)))
