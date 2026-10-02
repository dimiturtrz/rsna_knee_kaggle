"""Slices stored at a lower bit depth than their series, brought back onto the series' intensity scale.

The raw RSNA corpus holds two files whose header declares 16 bits allocated / 12 stored while the
pixel body is exactly one byte per pixel. Read as uint8 they are real images of their series —
correlation 0.934 and 0.761 with the neighbouring slice, whose own neighbour pairs read 0.938 and
0.889 — but on a darker scale (the neighbour fits about 1.3x and 1.5x the byte value). Dropping
them throws away a slice; keeping them raw serves one slice 25-30% darker than the stack it sits in,
under a percentile normalisation computed over the whole stack. So a sub-depth plane is matched to
its nearest full-depth plane by mean and spread, and kept only when it correlates with that plane
as an adjacent slice does.

Moment matching rather than a least-squares fit: regression shrinks the gain by the correlation,
which at r = 0.76 is a quarter of the scale this exists to restore.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from jaxtyping import Float
from numpy.typing import NDArray

MIN_CORRELATION = 0.7
"""Below both measured repairs (0.761, 0.934) and far above unrelated anatomy — a plane under it is
not shown to be this series' slice, so it is treated as a failed decode and filled from a neighbour."""

EIGHT_BITS = 8

DECODE_FAULTS = (OSError, ValueError, AttributeError, KeyError)


@dataclass(frozen=True)
class Plane:
    """One decoded slice in modality units, with the bit depth its pixels were actually stored at."""

    pixels: Float[NDArray[np.float32], "h w"]
    bits: int

    @classmethod
    def read(cls, path: Path) -> "Plane | None":
        """One slice, or None when it will not decode even as one byte per pixel."""
        try:
            data = pydicom.dcmread(str(path), force=True)
            pixels, bits = cls._stored(data)
        except DECODE_FAULTS:
            return None
        slope = float(getattr(data, "RescaleSlope", 1) or 1)
        intercept = float(getattr(data, "RescaleIntercept", 0) or 0)
        return cls(pixels * slope + intercept, bits)

    @staticmethod
    def _stored(data: pydicom.Dataset) -> tuple[Float[NDArray[np.float32], "h w"], int]:
        """The pixels as declared, or — when the body is one byte per pixel under a wider header — as bytes."""
        try:
            return data.pixel_array.astype(np.float32), int(getattr(data, "BitsStored", None) or data.BitsAllocated)
        except ValueError:
            body = data.PixelData
            if len(body) != data.Rows * data.Columns:
                raise
            return np.frombuffer(body, np.uint8).reshape(data.Rows, data.Columns).astype(np.float32), EIGHT_BITS


class DepthMatch:
    """A series' planes with every sub-depth one rescaled onto its full-depth neighbours, or dropped."""

    @classmethod
    def matched(cls, planes: list[Plane | None]) -> list[Float[NDArray[np.float32], "h w"] | None]:
        """The pixels of each plane; a sub-depth plane matched to its nearest full-depth one, or None."""
        full = max((plane.bits for plane in planes if plane is not None), default=0)
        references = [index for index, plane in enumerate(planes) if plane is not None and plane.bits == full]
        return [
            None if plane is None else cls._onto(plane, full, planes, references, index)
            for index, plane in enumerate(planes)
        ]

    @classmethod
    def _onto(
        cls, plane: Plane, full: int, planes: list[Plane | None], references: list[int], index: int
    ) -> Float[NDArray[np.float32], "h w"] | None:
        if plane.bits == full:
            return plane.pixels
        reference = planes[min(references, key=lambda other: abs(other - index))]
        return None if reference is None else cls.rescaled(plane.pixels, reference.pixels)

    @staticmethod
    def rescaled(
        pixels: Float[NDArray[np.float32], "h w"], reference: Float[NDArray[np.float32], "h w"]
    ) -> Float[NDArray[np.float32], "h w"] | None:
        """`pixels` on `reference`'s scale by mean and spread, or None when it is not shown to be its neighbour."""
        if pixels.shape != reference.shape or not pixels.std() or not reference.std():
            return None
        correlation = float(np.corrcoef(pixels.ravel(), reference.ravel())[0, 1])
        if correlation < MIN_CORRELATION:
            return None
        gain = reference.std() / pixels.std()
        return ((pixels - pixels.mean()) * gain + reference.mean()).astype(np.float32)
