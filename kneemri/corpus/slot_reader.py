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

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pydicom
import torch
from jaxtyping import UInt8
from numpy.typing import NDArray
from torch import Tensor, nn

from kneemri.corpus.acquisition import Acquisition
from kneemri.corpus.crop_centre import CROP_MM, Crop, CropCentre, ImageCentre

IMAGE_SIZE = 336
"""130 mm / 336 px = 0.387 mm per pixel, which clears the 0.5 mm a 1 mm tear needs."""

SLICE_BAND = (0.20, 0.80)
"""The central band the samples spread over: the outermost slices of a knee series are mostly soft
tissue outside the joint."""

GROUP = 3
"""How many slices one encoder feature covers — three, because a natural-image backbone takes three
channels and the cheapest way to give it depth is to stack a slice in each.

ONE home, because it is the train side, the recorded side and the serve side of a single number: the
trainer reads windows this wide, `member_record` stamps it into the config, and every reader windows a
member by what the config says. It was declared separately in all three (and a fourth, dead, copy in
`train`), so moving one would have fitted a member at one depth, stamped another and served a third —
with every fingerprint still passing, because each half is self-consistent."""


@dataclass(frozen=True)
class Band:
    """Which fraction of a series' depth the slices are drawn from, as a pair of fractions.

    A type rather than a tuple because it is a reading RULE, and every other reading rule in this
    package is one: it decides which slices exist to be sampled, it has to name the cache it produces,
    and it has to be stamped into a member's fingerprint as the run's own value. It was the last rule
    recorded from a module constant instead — `member_record` stamped `SLICE_BAND` while its neighbours
    each took the run's — which was harmless only because nothing could configure it. The serve path
    reads the recorded field, so the moment anything could, a member fitted on a wide band would have
    been served a narrow one with every fingerprint still passing.

    Both defaults are the historic reading, so a caller that says nothing gets the slices every cache on
    disk already holds.
    """

    low: float = SLICE_BAND[0]
    high: float = SLICE_BAND[1]

    def suffix(self) -> str:
        """What this band adds to a cache's name — empty for the historic one.

        Empty by design, for the reason `Crop.suffix` gives: those arrays are memory-mapped by running
        trainers and every member on disk was fitted on them, so the default must keep resolving to the
        file it always did. Any other band is a different set of pixels under the same shape, which is
        exactly the silent clobber the cache key exists to prevent.
        """
        return "" if (self.low, self.high) == SLICE_BAND else f"_{self.low:g}-{self.high:g}band"

    def recorded(self) -> list[float]:
        """The pair as a manifest writes it — JSON has no tuples, and every manifest holds a list."""
        return [self.low, self.high]

    @classmethod
    def of(cls, recorded: Sequence[float] | str) -> "Band":
        """The band a manifest recorded, read back — ours as a pair, a published one as text.

        Both spellings here rather than at the one call site that meets a published checkpoint, because
        this is the only place that knows what a recorded band looks like. Indexing a `"0.2,0.8"` string
        as a sequence reads the characters `0` and `.`, so the published spelling did not fail loudly:
        it reached `float(".")` and raised somewhere that had nothing to do with a band.
        """
        parts: Sequence[str] | Sequence[float] = recorded.split(",") if isinstance(recorded, str) else recorded
        return cls(low=float(parts[0]), high=float(parts[1]))


_PERCENTILES = (1.0, 99.0)

MIN_CROP_PIXELS = 16
"""Below this the crop is not a knee at any spacing, and the spacing is the thing to disbelieve."""


@dataclass(frozen=True)
class Reading:
    """The geometry a cache is decoded under: what is taken in plane, and what is taken through it.

    One object because the two are asked for together everywhere and because the cache's NAME is a
    function of the pair — `StudyCache` took them as separate arguments until the band arrived and
    pushed the constructor past the argument-count gate, which is the shape this repository has a
    strategy for. `Crop` was created from the same pressure one axis down.

    Both defaults are the historic reading, so a caller that says nothing resolves to the arrays every
    member on disk was fitted on.
    """

    crop: Crop = Crop()
    band: Band = Band()

    def suffix(self) -> str:
        """What this reading adds to a cache's name — empty for the historic one, on both axes."""
        return f"{self.crop.suffix()}{self.band.suffix()}"

    def reader(self, image_size: int) -> "SlotReader":
        """The reader that decodes under this geometry, at the size asked for.

        Here rather than at each call site because a caller assembling one reached through this object
        to the crop and then to the centre STRATEGY inside it — three deep, which is the coupling gate's
        complaint and a real one: every such site is a place a new axis has to be remembered.
        """
        return SlotReader(crop_mm=self.crop.mm, image_size=image_size, band=self.band, centre=self.crop.located())


class SlotReader:
    """Reads a series into `[slices, size, size]` uint8, physically scaled and intensity-normalised."""

    def __init__(
        self,
        crop_mm: float = CROP_MM,
        image_size: int = IMAGE_SIZE,
        band: Band | None = None,
        centre: CropCentre | None = None,
    ) -> None:
        self.crop_mm = crop_mm
        # Defaulting to the image centre keeps every cache on disk and every member fitted on one
        # byte-identical; a different centre is a different reading and is keyed as such.
        self.centre = centre or ImageCentre()
        self.image_size = image_size
        self.band = band or Band()

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
        low, high = int(self.band.low * (available - 1)), int(self.band.high * (available - 1))
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

    def window(
        self, stack: NDArray[np.float32], spacing: float | None, crop_mm: float | None = None
    ) -> tuple[slice, slice] | None:
        """The rows and columns a crop of `crop_mm` takes from this stack, or None when it does not apply.

        One home for the geometry because both crops here take it and a measurement that scores a box
        the reader would not take is measuring the wrong thing.

        None whenever the crop cannot be honoured: no spacing (the extent is not physical), or a window
        wider than the frame (the acquisition is already tighter than the crop, so cropping does
        nothing). Clamped INTO the frame rather than shortened, because a short crop is a different
        mm/pixel and that is the one property this whole read exists to hold constant.
        """
        wanted = self.crop_mm if crop_mm is None else crop_mm
        if spacing is None or not np.isfinite(spacing) or spacing <= 0:
            return None
        taken = round(wanted / spacing)
        height, width = stack.shape[1], stack.shape[2]
        if not MIN_CROP_PIXELS < taken < min(height, width):
            return None
        half = taken // 2
        centre = self.centre.of(stack)
        row = int(np.clip(centre[0], half, height - half))
        column = int(np.clip(centre[1], half, width - half))
        return slice(row - half, row + half), slice(column - half, column + half)

    def _crop(self, stack: NDArray[np.float32], spacing: float | None) -> NDArray[np.float32]:
        """Centre-crop to a constant physical extent, when the spacing is known and the crop fits."""
        taken = self.window(stack, spacing)
        return stack if taken is None else stack[:, taken[0], taken[1]]

    def narrowed(self, stack: NDArray[np.float32], spacing: float | None, crop_mm: float | None) -> NDArray[np.float32]:
        """A stack already cropped to this reader's extent, cropped again to a TIGHTER one.

        The multi-scale scheme reads the same series wide and tight, and the tight extent is a SUBSET of
        the wide one — so it is taken from the stack the wide read already produced rather than decoded
        again. Scale therefore costs encoder passes and not I/O, which is the whole reason it can be a
        slot axis instead of a second decode.

        None, or an extent no tighter than our own, passes the stack through untouched: a slot that
        names no crop is a plain sequence slot and must be byte-identical to what it always was.
        """
        if crop_mm is None or crop_mm >= self.crop_mm:
            return stack
        taken = self.window(stack, spacing, crop_mm)
        return stack if taken is None else stack[:, taken[0], taken[1]]

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
