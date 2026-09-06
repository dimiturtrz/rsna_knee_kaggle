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

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import override

import numpy as np
import pydicom
import torch
from jaxtyping import UInt8
from numpy.typing import NDArray
from torch import Tensor, nn

from core.data.corpus.acquisition import Acquisition
from core.data.corpus.crop_centre import CROP_MM, Crop, CropCentre, ImageCentre

IMAGE_SIZE = 336
"""130 mm / 336 px = 0.387 mm per pixel, which clears the 0.5 mm a 1 mm tear needs."""

SLICE_BAND = (0.20, 0.80)
"""The central band the samples spread over — the HISTORIC reading, and no longer the best one.

It is kept as the default because every cache on disk and every member fitted before 2026-08-18 was
built under it, and a default that moves rewrites what a caller who says nothing gets.

THE AXIS IS CLOSED AND THIS IS NOT ITS PEAK. Three points on the span axis at 12 slices: 0.20-0.80 =
0.8525, **0.10-0.90 = 0.8565**, 0.00-1.00 = 0.8427. An INTERIOR optimum — coverage does not keep
paying, and the outer tenths are active distraction rather than weak signal (the 0.00-1.00 arm fired
both halves of its pre-registered refutation, with Baker's, ACL and Contusion each losing more than
their own paired interval). So the band is a FILTER, not a budget, and narrowing past 0.20-0.80 walks
away from the peak rather than toward it.

**0.10-0.90 is the operating point every arm should read from, and every shipped member already does**
— it is passed explicitly (`--band-low 0.1 --band-high 0.9`), not inherited from here. Stated because
this constant reads like an unexamined guess and its neighbour in `band_coverage` said outright that
nothing had ever measured it; that sentence survived the measurement that refuted it and cost a later
reader an audit to rediscover a closed axis. See `interpretations/kneemri/2026-08-17_slice_band_coverage.md`.
"""

SPREAD = "spread"
"""Slices drawn evenly across the whole band — the historic rule, and what every cache on disk holds."""

ADJACENT = "adjacent"
"""A CONTIGUOUS run of slices centred in the band, so consecutive samples are consecutive in millimetres.

WHY THE AXIS EXISTS AT ALL. Width is EXTENT — how much of the depth can be seen — and spacing is
CONTIGUITY, which is a different mechanism and the one the deficit findings are defined by: a meniscal
tear is 1-3 mm of signal that has to touch an articular surface on two CONSECUTIVE slices. So the
interior optimum `SLICE_BAND` records (0.10-0.90 beats both 0.20-0.80 and 0.00-1.00) is not evidence
against this, and a competitor's +0.018 for a centre-adjacent nine over nine spread across twenty-four
is not evidence against the band result.

AND WHY IT IS NOW CLOSED, BY ARITHMETIC RATHER THAN BY AN ARM. This docstring used to assert that
"under `SPREAD` no two of the sampled slices are ever adjacent in the volume, AT ANY WIDTH". That is
false, and it is only true when the slice budget sits far below the in-band count. Series depth measured
over a 12-study sample (68 series): min 16, p25 26, median 32, p75 38. At `slices = 20`, which is the
operating point every current member is fitted at, the two rules very nearly COINCIDE — consecutive
pairs out of a possible 19, spread against adjacent:

    depth 16 (min)     13 in band     12 vs 12   — the two rules are IDENTICAL
    depth 26 (p25)     21 in band     18 vs 19
    depth 32 (median)  25 in band     14 vs 19
    depth 38 (p75)     31 in band      8 vs 19

So at the median study `SPREAD` already delivers 74% of the available contiguity and at p25 it delivers
95%, while `ADJACENT` still pays the full span cost (24 -> 19 indices at the median, 30 -> 19 at p75) —
the same span-for-contiguity trade that cost `adjacent224` its Regional findings. The contrast is large
only on deep series, which are exactly the ones where adjacency throws away the most span. There is no
operating point where the difference is both big and span-cheap, so the axis does not open by moving to
a denser stack; a denser stack is where `SPREAD` CONVERGES to `ADJACENT`. See `knee-e6sj`, whose
re-open condition this retires, and the 2026-09-05 ledger row.
"""


@dataclass(frozen=True)
class Spacing:
    """How the `slices` samples are laid out INSIDE the band — the rule, held in a table not a branch.

    A strategy for the reason `SlicePooling` and `TokenPooling` are: the vocabulary and the reduction
    live in one place, so a name that reaches a recorded config cannot fail to reach the consumer. That
    is this project's recurring plumbing defect, and it is worse here than anywhere else — a spacing
    that lands in a manifest but not in the decode would fit a member on one set of pixels and stamp
    another, with every fingerprint passing against its own reading.
    """

    def indices(self, low: int, high: int, slices: int) -> list[int]:
        """`(low, high, slices) -> indices`, both bounds inclusive and already clipped to the series."""
        raise NotImplementedError


class Spread(Spacing):
    """`linspace` across the band, de-duplicated then padded — what every member on disk was fitted on.

    THE DEDUPE IS WHAT CLUSTERS THE REPEATS, and it is left alone deliberately. Measured over a 60-study
    sample (315 series): 8.3% of series hold fewer than twelve slices in the band, and for those
    `linspace` alone would give evenly spread duplicates — [1,1,2,3,3,4,4,5,6,6,7,7] — while
    de-duplicating and then padding gives six copies of the deepest slice — [1,2,3,4,5,6,7,7,7,7,7,7].

    It is not fixed because the cost is wrong, not because it is right. Changing this changes the CACHE,
    so it invalidates 99 GB of decoded arrays and every member fitted on them, and no fingerprint would
    catch the difference. Against that: the result is identical under `max` pooling, which is
    duplicate-invariant, and the compute is identical either way. It skews only attention pooling, where
    six copies of one slice take six shares of the softmax.

    Fix it when a cache is next rebuilt for another reason, not on its own account.
    """

    @override
    def indices(self, low: int, high: int, slices: int) -> list[int]:
        """Evenly spread over the band, repeating the last when the band is thinner than asked."""
        return list(dict.fromkeys(np.linspace(low, high, slices).astype(int).tolist()))


class Adjacent(Spacing):
    """A contiguous run centred in the band, shifted rather than truncated when it will not fit.

    SHIFTED, NOT CLIPPED, because a run that hangs off one end and is cut there is neither contiguous nor
    centred — it is a shorter spread sample wearing this name, and it would fail silently on exactly the
    thin series `Spread`'s dedupe note measures at 8.3%. Sliding the window back inside the series keeps
    the one property the rule exists for; only a series holding fewer slices than asked can lose it, and
    that case pads like every other.
    """

    @override
    def indices(self, low: int, high: int, slices: int) -> list[int]:
        """`slices` consecutive indices about the band's midpoint, slid to stay inside `[low, high]`."""
        start = (low + high + 1 - slices) // 2
        return [index for index in range(max(low, min(start, high + 1 - slices)), high + 1)][:slices]


SPACINGS: dict[str, Spacing] = {SPREAD: Spread(), ADJACENT: Adjacent()}
"""The vocabulary, so a name is checkable at the flag and resolvable at the decode from one table."""

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
    spacing: str = SPREAD
    """How the samples sit inside the band. Here rather than on `Band` because `Band.recorded` is a pair
    of floats every manifest on disk already holds, and a third element of another type would change
    what a recorded band MEANS — this is a separate field with its own default, so every existing
    manifest keeps reading back as the reading it was written under."""

    def suffix(self) -> str:
        """What this reading adds to a cache's name — empty for the historic one, on all three axes."""
        return f"{self.crop.suffix()}{self.band.suffix()}{self.spacing_suffix()}"

    def spacing_suffix(self) -> str:
        """What the spacing adds to a cache's name — empty for `SPREAD`, for `Crop.suffix`'s reason.

        Those arrays are memory-mapped by running trainers and every member on disk was fitted on them,
        so the default must keep resolving to the file it always did. Any other spacing is a different
        set of pixels under the same shape, which is exactly the silent clobber the cache key prevents.
        """
        return "" if self.spacing == SPREAD else f"_{self.spacing}"

    def reader(self, image_size: int, memo: "FileMemo | None" = None) -> "SlotReader":
        """The reader that decodes under this geometry, at the size asked for.

        Here rather than at each call site because a caller assembling one reached through this object
        to the crop and then to the centre STRATEGY inside it — three deep, which is the coupling gate's
        complaint and a real one: every such site is a place a new axis has to be remembered.
        """
        return SlotReader(
            crop_mm=self.crop.mm,
            image_size=image_size,
            band=self.band,
            centre=self.crop.located(),
            spacing=self.spacing,
        ).sharing(memo)


@dataclass
class FileMemo:
    """ONE STUDY's decoded planes and file orders, so read groups that overlap open each file once.

    THE DUPLICATE IS ACROSS GROUPS, NOT INSIDE ONE. An entry's decode bill is the number of READ
    GROUPS (`Ensemble.reads`), and the champion's three groups — 20 slices at band 0.1-0.9, 12 at
    0.1-0.9, 12 at 0.2-0.8 — sample overlapping indices of the same series. Measured over the champion's
    groups on 96 studies, routed acquisitions only: 204.9 files opened per study against 105.6 distinct,
    so 48.5% of the opens are the same file decoded again. The header parse duplicates outright —
    `ordered_files` reads every file in a series to sort it, and its answer cannot depend on the band.

    IT IS PER STUDY AND MUST STAY PER STUDY. Series names are unique inside a study and nowhere else,
    and the whole corpus's planes do not fit in memory (about 8 MB a study). One memo is built per
    `decoded_at` call, lives inside the worker process for that study, and dies with it.

    Nothing here changes a pixel: a memoised plane IS the array the second decode would have produced,
    so every member is handed byte-identical bags. That equality is the change's admission ticket.
    """

    planes: dict[tuple[str, str], NDArray[np.float32] | None] = field(default_factory=dict)
    orders: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def ordered(self, acquisition: Acquisition) -> tuple[str, ...]:
        """This series' file order, header-parsed once however many groups ask for it."""
        if acquisition.series not in self.orders:
            self.orders[acquisition.series] = acquisition.ordered_files()
        return self.orders[acquisition.series]

    def plane(self, acquisition: Acquisition, name: str, decoded: NDArray[np.float32] | None) -> None:
        """Keep a decode, including a FAILED one — a file that will not open must not be retried."""
        self.planes[(acquisition.series, name)] = decoded

    def opened(self, acquisition: Acquisition, name: str) -> bool:
        """Whether this file has already been decoded under some other group."""
        return (acquisition.series, name) in self.planes


class SlotReader:
    """Reads a series into `[slices, size, size]` uint8, physically scaled and intensity-normalised."""

    def __init__(
        self,
        crop_mm: float = CROP_MM,
        image_size: int = IMAGE_SIZE,
        band: Band | None = None,
        centre: CropCentre | None = None,
        spacing: str = SPREAD,
    ) -> None:
        self.memo: FileMemo | None = None
        self.crop_mm = crop_mm
        self.spacing = SPACINGS[spacing]
        # Defaulting to the image centre keeps every cache on disk and every member fitted on one
        # byte-identical; a different centre is a different reading and is keyed as such.
        self.centre = centre or ImageCentre()
        self.image_size = image_size
        self.band = band or Band()

    def sharing(self, memo: "FileMemo | None") -> "SlotReader":
        """Route this reader's file opens through a memo, and hand it back.

        Not a constructor argument because it is not part of the READING: two readers differing only in
        their memo decode byte-identical pixels, and the geometry constructor is already at the argument
        ceiling with the five knobs that DO change what comes back.
        """
        self.memo = memo
        return self

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

        Or None when the series has no files, or when nothing in the sampled band decodes — see
        `_stacked` for what "nothing decodes" is allowed to mean and why it is not a black stack.
        """
        ordered = acquisition.ordered_files() if self.memo is None else self.memo.ordered(acquisition)
        if not ordered:
            return None
        planes = [self._decoded(acquisition, ordered[index]) for index in self.band_indices(len(ordered), slices)]
        stack = self._stacked(planes)
        return None if stack is None else self._crop(stack, acquisition.pixel_spacing)

    @classmethod
    def _stacked(cls, planes: list[NDArray[np.float32] | None]) -> NDArray[np.float32] | None:
        """The sampled planes as one array — failures filled from a neighbour, odd matrix sizes dropped.

        None when NOTHING decoded: such a series is reported ABSENT, which the presence mask can
        express, rather than black, which it cannot. A single failed slice is filled from the nearest
        slice that did decode — never from a zero plane, which would otherwise become a candidate for
        the stack's shape and blacken every good slice with it.

        Slices of one series can still differ in matrix size — multi-echo and some reformats do — and
        those are genuinely not stackable, so an odd one out is dropped to zeros rather than resized.
        THE MAJORITY SHAPE DECIDES, not the first sampled slice. With the first plane as authority, a
        series whose FIRST sampled slice is the odd one out blackened every other slice — eleven good
        planes zeroed by one — and the presence mask still called the slot acquired, so the model was
        served a black slot as if it were real. Which slice the band happens to sample first is not a
        property of the series. Measured worth on THIS corpus: zero — `diagnose_matrix` found 0 mixed
        series in 1682 over 300 studies, so this is correctness insurance for the hidden test, not a
        score change to expect.
        """
        decoded = [index for index, plane in enumerate(planes) if plane is not None]
        if not decoded:
            return None
        filled: list[NDArray[np.float32]] = [
            plane if plane is not None else cls._nearest(planes, decoded, index) for index, plane in enumerate(planes)
        ]
        shape = Counter(plane.shape for plane in filled).most_common(1)[0][0]
        return np.stack([plane if plane.shape == shape else np.zeros(shape, np.float32) for plane in filled])

    @staticmethod
    def _nearest(planes: list[NDArray[np.float32] | None], decoded: list[int], index: int) -> NDArray[np.float32]:
        """The decoded slice closest to `index` — a failure is filled from its neighbour, never zeros."""
        nearest = planes[min(decoded, key=lambda other: abs(other - index))]
        if nearest is None:
            raise ValueError("decoded indices must point at decoded planes")
        return nearest

    def band_indices(self, available: int, slices: int) -> list[int]:
        """Indices inside the central band under this reader's spacing, padded when the band is thin.

        PUBLIC because the band is a reading rule and not a DICOM one. A second corpus arrives as whole
        pickled volumes rather than as files to decode lazily, so it cannot enter through `cropped` — but
        it must be sampled through the depth by the SAME rule, or the two corpora differ on an axis that
        is measured (0.20-0.80 = 0.8525, 0.10-0.90 = 0.8565, 0.00-1.00 = 0.8427) while claiming not to.

        The bounds and the padding are shared by every spacing; only the layout between them is the
        strategy's, which is why the `Spacing` table returns indices and this method finishes them.
        """
        low, high = int(self.band.low * (available - 1)), int(self.band.high * (available - 1))
        if high <= low:
            return [available // 2] * slices
        indices = self.spacing.indices(low, high, slices)
        while len(indices) < slices:
            indices.append(indices[-1])
        return indices[:slices]

    def _decoded(self, acquisition: Acquisition, name: str) -> NDArray[np.float32] | None:
        """One slice, through this reader's memo when it has one — the same array either way."""
        if self.memo is None:
            return self._decode(acquisition, name)
        if not self.memo.opened(acquisition, name):
            self.memo.plane(acquisition, name, self._decode(acquisition, name))
        return self.memo.planes[(acquisition.series, name)]

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
