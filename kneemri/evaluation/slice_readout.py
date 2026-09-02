"""How the preserved slice axis is reduced before a head or a probe sees it.

A strategy because the reduction is the instrument's own choice, not the trunk's, and it was the
unmeasured half of every trunk number this project has quoted. `--slice-read` names a readout in the
cache FILENAME but never reached the encoder, so `..._focal_max.npy` and `..._focal_attend.npy` are the
same bytes (knee-n647): the slice axis survives in the cache and is thrown away at read time instead.
"""

from typing import Protocol

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray


class SliceReadout(Protocol):
    """The contract a readout answers: slots and slices in, one vector per study out."""

    name: str

    def reduce(
        self, array: Float[NDArray[np.floating], "studies slots slices dim"]
    ) -> Float[NDArray[np.float64], "studies width"]:
        """Collapse slots and slices to the one vector a head is handed."""
        ...


class SliceMean:
    """The average over slots and slices — what every trunk number on the ledger was read with."""

    name = "mean"

    def reduce(
        self, array: Float[NDArray[np.floating], "studies slots slices dim"]
    ) -> Float[NDArray[np.float64], "studies width"]:
        """Average away both axes, the reduction every slot-pooling head starts from."""
        return np.asarray(array, dtype=np.float64).mean(axis=(1, 2))


class SliceMax:
    """The per-channel peak along the stack, which is what a whole-stack max head reads."""

    name = "max"

    def reduce(
        self, array: Float[NDArray[np.floating], "studies slots slices dim"]
    ) -> Float[NDArray[np.float64], "studies width"]:
        """Take the maximum over slices before averaging slots, so one slice can carry a channel."""
        return np.asarray(array, dtype=np.float64).max(axis=2).mean(axis=1)


class SliceMoments:
    """The mean along the stack beside its spread, at twice the width.

    The question a learned query is expensive to ask cheaply: does HOW a channel varies along the stack
    carry anything the average discards. A concatenation over all twelve slices would answer it directly
    and cannot be afforded — twelve times 3072 columns makes a ridge gram of about 11 GB — so the second
    moment stands in for the axis at 2x width instead of 12x.
    """

    name = "moments"

    def reduce(
        self, array: Float[NDArray[np.floating], "studies slots slices dim"]
    ) -> Float[NDArray[np.float64], "studies width"]:
        """Average and standard deviation along the stack, concatenated."""
        dense = np.asarray(array, dtype=np.float64)
        return np.hstack([dense.mean(axis=(1, 2)), dense.std(axis=2).mean(axis=1)])


READOUTS: dict[str, SliceReadout] = {readout.name: readout for readout in (SliceMean(), SliceMax(), SliceMoments())}
"""The slice readouts a probe may be run under, keyed by the name `--slice-readout` takes."""
