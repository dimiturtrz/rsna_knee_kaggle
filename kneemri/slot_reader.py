"""Turning one series into the fixed-shape uint8 stack the encoder eats.

Two normalisations happen here, and both are properties of the resize rather than of the network:

**Physical scale.** A slice of N pixels at spacing s covers Ns mm, and both vary across this corpus
(spacing median 0.332 mm here, range several-fold). Cropping to a constant physical extent L and
resampling to P pixels fixes the pitch at L/P mm per pixel. A feature of width d survives only if the
pitch is at most d/2 — Nyquist on the resampling grid — and a meniscal tear is 1-3 mm, so 1 mm demands
0.5 mm/pixel. At L=130 mm, P=224 gives 0.580 (fails) and P=336 gives 0.387 (clears). The crop must be
SMALLER than the field of view or it silently does nothing.

**Intensity.** MR has no absolute scale. Each series is normalised to its own 1st-99th percentile, taken
over the sampled stack rather than per slice so slices keep their relative contrast, and by percentile
rather than extremes so one bright vessel does not compress everything else.
"""

import numpy as np
import pydicom
import torch
from jaxtyping import UInt8
from numpy.typing import NDArray
from torch import Tensor, nn

from kneemri.acquisition import Acquisition
from kneemri.crop_centre import CROP_MM, CropCentre, ImageCentre

IMAGE_SIZE = 336
"""130 mm / 336 px = 0.387 mm per pixel, which clears the 0.5 mm a 1 mm tear needs."""

SLICE_BAND = (0.20, 0.80)
"""The central band the samples spread over: the outermost slices of a knee series are mostly soft
tissue outside the joint."""

_MIN_CROP_PIXELS = 16
"""Below this the crop is nonsense rather than a crop, so the series passes through uncropped."""

_PERCENTILES = (1.0, 99.0)


class SlotReader:
    """Reads a series into `[slices, size, size]` uint8, physically scaled and intensity-normalised."""

    def __init__(
        self,
        crop_mm: float = CROP_MM,
        image_size: int = IMAGE_SIZE,
        band: tuple[float, float] = SLICE_BAND,
        centre: CropCentre | None = None,
    ) -> None:
        self.crop_mm = crop_mm
        # Defaulting to the image centre keeps every cache on disk and every member fitted on one
        # byte-identical; a different centre is a different reading and is keyed as such.
        self.centre = centre or ImageCentre()
        self.image_size = image_size
        self.band = band

    def read(self, acquisition: Acquisition, slices: int) -> UInt8[Tensor, "s h w"] | None:
        """Sample `slices` physically-spread slices at this reader's size, or None if nothing decodes."""
        cropped = self.cropped(acquisition, slices)
        return None if cropped is None else self.sized(cropped, self.image_size)

    def cropped(self, acquisition: Acquisition, slices: int) -> NDArray[np.float32] | None:
        """The decoded, physically-cropped stack — everything expensive, and NOTHING size-dependent.

        Split out because the same series is currently decoded once PER PIXEL GROUP: a submission
        carrying members that read 392, 336 and 224 px reads every DICOM three times, and decode is the
        I/O-bound half on a Kaggle host with four CPUs. Reading once and resizing per target is exact
        rather than approximate — each size still comes from ONE bilinear resize of this same stack, so
        the bytes are identical to a native read and no fingerprint or witness moves.

        The alternative considered and rejected: decode at the largest size and downsample between
        sizes. Measured at about one grey level in 255 against a native decode — small, but a needless
        change to members that already score.

        Or None when nothing in the series decodes. A series where nothing decodes is reported ABSENT,
        which the presence mask can express, rather than black, which it cannot. A single failed slice is
        filled from the nearest slice that did decode — never from a zero plane, which would otherwise
        become the authority for the stack's shape and blacken every good slice with it.

        A series where nothing decodes is reported ABSENT, which the presence mask can express, rather
        than black, which it cannot. A single failed slice is filled from the nearest slice that did
        decode — never from a zero plane, which would otherwise become the authority for the stack's
        shape and blacken every good slice with it.
        """
        ordered = acquisition.ordered_files()
        if not ordered:
            return None
        planes = [self._decode(acquisition, ordered[index]) for index in self._sample_indices(len(ordered), slices)]
        decoded = [index for index, plane in enumerate(planes) if plane is not None]
        if not decoded:
            return None
        filled: list[NDArray[np.float32]] = [
            plane if plane is not None else self._nearest(planes, decoded, index) for index, plane in enumerate(planes)
        ]
        # Slices of one series can still differ in matrix size — multi-echo and some reformats do — and
        # those are genuinely not stackable, so an odd one out is dropped to zeros rather than resized.
        shape = filled[0].shape
        stack = np.stack([plane if plane.shape == shape else np.zeros(shape, np.float32) for plane in filled])
        return self._crop(stack, acquisition.pixel_spacing)

    @staticmethod
    def _nearest(planes: list[NDArray[np.float32] | None], decoded: list[int], index: int) -> NDArray[np.float32]:
        """The decoded slice closest to `index` — a failure is filled from its neighbour, never zeros."""
        nearest = planes[min(decoded, key=lambda other: abs(other - index))]
        if nearest is None:
            raise ValueError("decoded indices must point at decoded planes")
        return nearest

    def _sample_indices(self, available: int, slices: int) -> list[int]:
        """Indices spread over the central band, repeating the last when the band is thinner than asked.

        THE DEDUPE IS WHAT CLUSTERS THE REPEATS, and it is left alone deliberately. Measured over a
        60-study sample (315 series): 8.3% of series hold fewer than twelve slices in the band, and for
        those `linspace` alone would give evenly spread duplicates — [1,1,2,3,3,4,4,5,6,6,7,7] — while
        de-duplicating and then padding gives six copies of the deepest slice — [1,2,3,4,5,6,7,7,7,7,7,7].

        It is not fixed because the cost is wrong, not because it is right. Changing this changes the
        CACHE, so it invalidates 99 GB of decoded arrays and every member fitted on them, and no
        fingerprint would catch the difference. Against that: the result is identical under `max`
        pooling, which is duplicate-invariant, and the compute is identical either way. It skews only
        attention pooling, where six copies of one slice take six shares of the softmax.

        Fix it when a cache is next rebuilt for another reason, not on its own account.
        """
        low, high = int(self.band[0] * (available - 1)), int(self.band[1] * (available - 1))
        if high <= low:
            return [available // 2] * slices
        indices = list(dict.fromkeys(np.linspace(low, high, slices).astype(int).tolist()))
        while len(indices) < slices:
            indices.append(indices[-1])
        return indices[:slices]

    @staticmethod
    def _decode(acquisition: Acquisition, name: str) -> NDArray[np.float32] | None:
        """One slice in modality units, or None when it will not decode."""
        try:
            data = pydicom.dcmread(str(acquisition.directory / name), force=True)
            pixels = data.pixel_array.astype(np.float32)
        except (OSError, ValueError, AttributeError, KeyError):
            return None
        slope = float(getattr(data, "RescaleSlope", 1) or 1)
        intercept = float(getattr(data, "RescaleIntercept", 0) or 0)
        return pixels * slope + intercept

    def _crop(self, stack: NDArray[np.float32], spacing: float | None) -> NDArray[np.float32]:
        """Centre-crop to a constant physical extent, when the spacing is known and the crop fits."""
        if spacing is None or not np.isfinite(spacing) or spacing <= 0:
            return stack
        wanted = round(self.crop_mm / spacing)
        height, width = stack.shape[1:]
        if not _MIN_CROP_PIXELS < wanted < min(height, width):
            return stack
        half = wanted // 2
        centre_y, centre_x = self.centre.of(stack)
        # CLAMPED INTO THE FRAME, so a centre near an edge still yields the full physical extent rather
        # than a crop silently short on one side — which would be a different mm/pixel for that study.
        centre_y = int(np.clip(centre_y, half, height - half))
        centre_x = int(np.clip(centre_x, half, width - half))
        return stack[:, max(0, centre_y - half) : centre_y + half, max(0, centre_x - half) : centre_x + half]

    def narrowed(self, stack: NDArray[np.float32], spacing: float | None, crop_mm: float | None) -> NDArray[np.float32]:
        """A stack already cropped to this reader's extent, cropped again to a TIGHTER one.

        The multi-scale scheme reads the same series wide and tight, and the tight extent is a SUBSET of
        the wide one — so it is taken from the stack the wide read already produced rather than decoded
        again. Scale therefore costs encoder passes and not I/O, which is the whole reason it can be a
        slot axis instead of a second decode.

        None, or an extent no tighter than our own, passes the stack through untouched: a slot that
        names no crop is a plain sequence slot and must be byte-identical to what it always was.
        """
        if crop_mm is None or crop_mm >= self.crop_mm or spacing is None or not np.isfinite(spacing) or spacing <= 0:
            return stack
        wanted = round(crop_mm / spacing)
        height, width = stack.shape[1:]
        if not _MIN_CROP_PIXELS < wanted < min(height, width):
            return stack
        half = wanted // 2
        centre_y, centre_x = self.centre.of(stack)
        centre_y = int(np.clip(centre_y, half, height - half))
        centre_x = int(np.clip(centre_x, half, width - half))
        return stack[:, centre_y - half : centre_y + half, centre_x - half : centre_x + half]

    def sized(self, stack: NDArray[np.float32], size: int) -> UInt8[Tensor, "s h w"]:
        """Percentile-normalise, resize to the encoder's grid, and store eight bits per pixel.

        uint8 rather than float32 because these buffers are the largest allocation in the pipeline and
        the intensity is already in [0, 1] — eight bits cost nothing a bilinear resize has not.
        """
        low, high = np.percentile(stack, _PERCENTILES)
        scaled = np.clip((stack - low) / max(high - low, 1e-6), 0, 1)
        tensor = torch.from_numpy(np.ascontiguousarray(scaled)).unsqueeze(0)
        resized = nn.functional.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
        return (resized.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)
