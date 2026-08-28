"""The raptor decode, reimplemented as OUR code from its published description.

`python -m kneemri.evaluation.raptor_feed --studies 12` — decode the same studies through their module
and through this one, and assert the two volumes are equal element for element.

WHY A REIMPLEMENTATION AND NOT THE IMPORT WE ALREADY HAVE. `foreign_member` imports their script by
path, which is right for measuring on our own disk and useless for shipping: `external/raptor` is
gitignored third-party code with no LICENSE file, so it cannot go into a Kaggle kit. Their notebook,
however, publishes the whole specification in prose — five fixed slots, the 6–94% span, the millimetre
crop, the percentile window, the three-slice stacking, the window count. Writing that spec out in our
own code is not vendoring, and it is what lets the member be SERVED.

THE SECOND PAYOFF IS THE BIGGER ONE. `2026-08-27_the-feed-was-the-whole-shortfall.md` measured the
byte-identical checkpoint at −0.0348 through our feed and +0.030 above our best member through theirs.
The shortfall was the feed, not the weights. Once their feed is our code it stops being a serving
adapter and becomes a candidate TRAINING feed — the axis that measurement says is worth the most, and
the one we presently cannot test at all.

FAITHFUL MEANS BYTE-IDENTICAL, NOT CLOSE. Both decodes turn the same DICOM files into a `uint8` volume,
so there is no tolerance to argue about: `--studies N` compares ours against theirs and any difference
at all is a bug here. A decode that is nearly right is the failure mode this whole file exists to avoid,
because it would score as a slightly weak member rather than as an error.

WHERE IT KNOWINGLY DIFFERS, AND WHY THAT IS SAFE. Their loops answer a broken DICOM with a bare
`except Exception`. This one names what it catches (`DECODE_FAULTS`), because a blanket catch is the
gate this repo does not suspend for convenience. The equivalence run over the holdout is what proves
the narrower catch loses nothing: the same 909 studies already decode at 0.0% fallback through their
path, so no study reaches those handlers at all on our disk.
"""

import csv
import glob
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from pydicom.errors import InvalidDicomError
from pydicom.pixel_data_handlers.util import apply_modality_lut
from torch import Tensor

from core import cli
from kneemri.evaluation import foreign_member

logger = logging.getLogger(__name__)

Records = Mapping[str, Sequence[Mapping[str, object]]]
"""The series rows of each study, keyed by study UID — the same shape `foreign_member` reads with
`csv.DictReader`, so a feed can be swapped under it without touching how the rows are loaded."""

IMAGE_SIZE = 336
"""Pixels on a side after the millimetre crop is resized. Fixed by the weights, not a preference."""

CROP_MM = 140.0
"""The side of the centre box, in MILLIMETRES rather than pixels.

This is the part of their decode that does real work: cropping by physical size means a knee occupies
the same fraction of the frame whether the study was acquired at 0.3 or 0.5 mm per pixel, so the model
is never asked to learn a scale difference that carries no medical information."""

MINIMUM_SPACING_MM = 1e-3
"""Floor on the divisor when pixel spacing is missing or absurd, so a bad header cannot divide by zero."""

DEFAULT_SPACING_MM = 0.5
"""The spacing assumed for a slice whose header does not carry `PixelSpacing`."""


@dataclass(frozen=True)
class Slot:
    """One of the five fixed positions in the stack: which plane, which weighting, how many slices."""

    plane: str
    fluid_sensitive: int
    slices: int


ANY_WEIGHTING = -1
"""`Slot.fluid_sensitive` for the axial slot, which takes whatever series the study has."""

SLOTS = (
    Slot("Sagittal", 1, 18),
    Slot("Sagittal", 0, 14),
    Slot("Coronal", 1, 12),
    Slot("Coronal", 0, 8),
    Slot("Axial", ANY_WEIGHTING, 12),
)
"""The stack, always in this order.

Preferring a fluid-sensitive series for some slots and not for others is deliberate and is theirs:
fluid-sensitive sequences show swelling, effusion and acute injury, the others show anatomy and
cartilage, and the twelve findings are split across both. A study missing a slot leaves it as zeros and
the mask tells the model to skip it, rather than being fed something misleading."""

STACK_DEPTH = sum(slot.slices for slot in SLOTS)
"""64 — the depth every study is padded or truncated to."""

SPAN_LOW = 0.06
SPAN_HIGH = 0.94
"""The fraction of a series the slices are drawn across.

Wide on purpose: the collateral ligaments and the lateral meniscus live in the peripheral slices that a
0.15–0.85 band throws away, and cutting them was measurably costing accuracy on exactly those findings.
Must match the corpus the weights were trained on."""

WINDOW_PERCENTILES = (2.0, 98.0)
"""The intensity window, taken over the whole SERIES rather than per slice — one window per slot keeps
slices within a series comparable to each other."""

BYTE_MAX = 255
"""The volume is stored as `uint8`; this is the scale factor into it and back out again."""

EVAL_WINDOWS = 42
"""How many three-slice windows a study is scored on. Their `K_EVAL`."""

TRIPLET = 3
"""A window is the slice and its two neighbours, stacked into the three channels of one image."""

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(TRIPLET, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(TRIPLET, 1, 1)

DECODE_FAULTS = (
    InvalidDicomError,
    OSError,
    ValueError,
    AttributeError,
    KeyError,
    IndexError,
    TypeError,
)
"""What a broken DICOM is allowed to raise before the slice is dropped rather than the study lost.

Named rather than blanket, and `InvalidDicomError` is named FIRST because it is NOT an `OSError` — it
derives straight from `Exception`, so a file that is not DICOM at all escaped an otherwise sensible
list. The unit test that writes sixteen bytes of junk into a series directory is what found that.

A decompression backend that is simply absent raises `RuntimeError` and is deliberately NOT caught:
that is a missing dependency rather than a bad file, and silently zeroing a slice for it is how a
member gets priced against a broken environment instead of against its weights."""


@dataclass(frozen=True)
class Slice:
    """One DICOM file, with the two header facts the decode needs from it."""

    position: float
    path: str
    spacing_mm: float


@dataclass(frozen=True)
class Series:
    """The slices of one series, ordered head to foot, plus the spacing to fall back on."""

    slices: tuple[Slice, ...]
    median_spacing_mm: float

    @staticmethod
    def read(directory: str) -> "Series":
        """Order a series directory by true anatomical position, not by filename.

        The projection of `ImagePositionPatient` onto the slice normal is the only ordering that is
        right for an oblique acquisition; `InstanceNumber` is the fallback for a header that does not
        carry the geometry, and is what most series would sort by anyway.
        """
        found: list[Slice] = []
        spacings: list[float] = []
        for path in glob.glob(directory + "/*.dcm"):
            try:
                header = pydicom.dcmread(path, stop_before_pixels=True)
                spacing = Series._spacing(header)
                found.append(Slice(Series._position(header), path, spacing))
                spacings.append(spacing)
            except DECODE_FAULTS:
                logger.debug("unreadable header, ordering it first: %s", path)
                found.append(Slice(0.0, path, DEFAULT_SPACING_MM))
        found.sort(key=lambda one: one.position)
        median = float(np.median(spacings)) if spacings else DEFAULT_SPACING_MM
        return Series(tuple(found), median)

    @staticmethod
    def _position(header: pydicom.Dataset) -> float:
        """Where this slice sits along the normal to its own plane."""
        orientation = getattr(header, "ImageOrientationPatient", None)
        origin = getattr(header, "ImagePositionPatient", None)
        oriented = orientation is not None and origin is not None and len(orientation) == TRIPLET * 2
        if not oriented:
            return float(getattr(header, "InstanceNumber", 0) or 0)
        row = np.array(orientation[:TRIPLET], float)
        column = np.array(orientation[TRIPLET:], float)
        return float(np.dot(np.array(origin, float), np.cross(row, column)))

    @staticmethod
    def _spacing(header: pydicom.Dataset) -> float:
        """Millimetres per pixel, defaulted when the header does not say."""
        spacing = getattr(header, "PixelSpacing", None)
        return float(spacing[0]) if spacing is not None else DEFAULT_SPACING_MM

    def sampled(self, count: int) -> tuple[Slice, ...]:
        """Take `count` slices evenly across the span, repeating the first if there is only one."""
        total = len(self.slices)
        if total <= 1:
            return tuple(self.slices[0] for _ in range(count)) if total else ()
        low = int(total * SPAN_LOW)
        high = max(int(total * SPAN_HIGH) - 1, low)
        picks = np.linspace(low, high, count).round().astype(int)
        return tuple(self.slices[min(int(pick), total - 1)] for pick in picks)


@dataclass(frozen=True)
class RaptorFeed:
    """Their study decode, written out here: DICOM directory in, a scorable `uint8` volume out."""

    series_root: Path

    @staticmethod
    def records(series_csv: Path) -> Records:
        """Their `ser_records`, read with `csv` so every key and value stays a `str`."""
        rows: dict[str, list[Mapping[str, object]]] = {}
        with series_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.setdefault(row["StudyInstanceUID"], []).append(row)
        return rows

    def study(
        self, study: str, records: Records
    ) -> tuple[UInt8[NDArray[np.uint8], "depth height width"], UInt8[NDArray[np.uint8], " depth"]]:
        """Fill the five slots for one study and report which depths actually hold an image."""
        rows = list(records.get(study, []))
        volume = np.zeros((STACK_DEPTH, IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        depth = 0
        used: set[object] = set()
        for slot in SLOTS:
            chosen = RaptorFeed._chosen(rows, slot, used)
            if chosen is None:
                depth += slot.slices
                continue
            used.add(chosen["SeriesInstanceUID"])
            series = Series.read(f"{self.series_root}/{study}/{chosen['SeriesInstanceUID']}")
            if not series.slices:
                depth += slot.slices
                continue
            depth = self._filled(volume, depth, series, slot)
            if depth >= STACK_DEPTH:
                break
        mask = (volume.reshape(STACK_DEPTH, -1).sum(1) > 0).astype(np.uint8)
        return volume, mask

    @staticmethod
    def _chosen(rows: Sequence[Mapping[str, object]], slot: Slot, used: set[object]) -> Mapping[str, object] | None:
        """The series for one slot: right plane, not already spent, preferring the asked weighting.

        Preference, not requirement — a study with no fluid-sensitive sagittal still fills the slot
        with the sagittal it has, because a zeroed slot costs more than a mis-weighted one.
        """
        candidates = [
            row for row in rows if row["Anatomical_Plane"] == slot.plane and row["SeriesInstanceUID"] not in used
        ]
        if slot.fluid_sensitive != ANY_WEIGHTING:
            preferred = [
                row for row in candidates if int(str(row.get("Fluid_Sensitive", 0) or 0)) == slot.fluid_sensitive
            ]
            if preferred:
                return preferred[0]
        return candidates[0] if candidates else None

    def _filled(
        self,
        volume: UInt8[NDArray[np.uint8], "depth height width"],
        depth: int,
        series: Series,
        slot: Slot,
    ) -> int:
        """Write one slot's slices into the volume and hand back the depth to continue at."""
        picked = series.sampled(slot.slices)
        images = [RaptorFeed._pixels(one.path) for one in picked]
        low, high = RaptorFeed._window(images)
        for image, one in zip(images, picked, strict=True):
            if depth >= STACK_DEPTH:
                break
            if image is None:
                depth += 1
                continue
            windowed = np.clip((image - low) / (high - low + 1e-6), 0, 1)
            spacing = one.spacing_mm if one.spacing_mm > 0 else series.median_spacing_mm
            volume[depth] = (RaptorFeed._cropped(windowed, spacing) * BYTE_MAX).astype(np.uint8)
            depth += 1
        return depth

    @staticmethod
    def _pixels(path: str) -> Float[NDArray[np.float32], "height width"] | None:
        """Read one slice into modality units, inverted if the study stores white-on-black."""
        try:
            slice_data = pydicom.dcmread(path)
            pixels = apply_modality_lut(slice_data.pixel_array, slice_data).astype(np.float32)
        except DECODE_FAULTS:
            logger.debug("unreadable pixels, leaving the depth blank: %s", path)
            return None
        if str(getattr(slice_data, "PhotometricInterpretation", "")) == "MONOCHROME1":
            return np.asarray(pixels.max() - pixels, np.float32)
        return pixels

    @staticmethod
    def _window(
        images: Sequence[Float[NDArray[np.float32], "height width"] | None],
    ) -> tuple[np.float64, np.float64]:
        """The intensity window for a whole slot, over every slice of it that read.

        RETURNED AS `np.float64`, NOT AS `float`, AND THAT IS LOAD-BEARING. Under NEP 50 a Python float
        is a WEAK scalar, so `float32_image - float(low)` stays float32, while a NumPy scalar keeps its
        own width and promotes the subtraction to float64. Their decode windows in float64 because
        `np.percentile` hands back `np.float64`; casting here moved the whole crop into float32 and
        shifted one pixel in 1,400 by a single count out of 255. Nothing about that is visible in a
        score, which is exactly why the equality oracle exists.
        """
        readable = [image for image in images if image is not None]
        if not readable:
            return np.float64(0.0), np.float64(1.0)
        low, high = np.percentile(np.concatenate([image.ravel() for image in readable]), WINDOW_PERCENTILES)
        return np.float64(low), np.float64(high)

    @staticmethod
    def _cropped(
        image: Float[NDArray[np.float64], "height width"], spacing_mm: float
    ) -> Float[NDArray[np.float64], "size size"]:
        """Centre-crop a fixed number of MILLIMETRES, then resize to the model's input."""
        height, width = image.shape
        side = min(round(CROP_MM / max(spacing_mm, MINIMUM_SPACING_MM)), height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        cropped = image[top : top + side, left : left + side]
        resized = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        return np.asarray(resized, dtype=np.float64)


@dataclass(frozen=True)
class Windows:
    """The scoring view of a decoded study: every three-slice window the mask allows."""

    count: int = EVAL_WINDOWS

    def centres(self, mask: UInt8[NDArray[np.uint8], " depth"]) -> list[int]:
        """The slice indices to centre windows on — inside the filled span, never on its edge."""
        depth = int(mask.shape[0])
        filled = np.where(mask > 0)[0]
        if len(filled) < TRIPLET:
            filled = np.arange(min(TRIPLET, depth))
        low, high = int(filled.min()), int(filled.max())
        inside = list(range(low + 1, high))
        if not inside:
            inside = [max(1, min((low + high) // 2, depth - 2))]
        picks = np.linspace(0, len(inside) - 1, self.count).round().astype(int)
        return [inside[int(pick)] for pick in picks]

    def stacked(
        self,
        volume: UInt8[NDArray[np.uint8], "depth height width"],
        mask: UInt8[NDArray[np.uint8], " depth"],
        resolution: int,
    ) -> Float[Tensor, "windows channels size size"]:
        """Stack each centre with its two neighbours and normalise the way the weights expect."""
        depth = volume.shape[0]
        centres = self.centres(mask)
        windows = np.empty((len(centres), TRIPLET, resolution, resolution), np.float32)
        for position, centre in enumerate(centres):
            middle = max(1, min(centre, depth - 2))
            triplet = np.stack([volume[middle - 1], volume[middle], volume[middle + 1]], 0, dtype=np.float32)
            windows[position] = Windows._resized(torch.from_numpy(triplet / BYTE_MAX), resolution).numpy()
        return (torch.from_numpy(windows) - IMAGENET_MEAN) / IMAGENET_STD

    @staticmethod
    def _resized(
        triplet: Float[Tensor, "channels height width"], resolution: int
    ) -> Float[Tensor, "channels size size"]:
        """Bring one window to the model's resolution, left alone when it is already there."""
        if triplet.shape[-1] == resolution:
            return triplet
        return F.interpolate(triplet[None], size=(resolution, resolution), mode="bilinear", align_corners=False)[0]


def main() -> None:
    """Decode the same studies both ways and refuse anything short of equality."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--studies", type=int, default=12, help="how many studies to compare")
    arguments = command.parsed()
    raw = command.rooted(arguments).raw()

    theirs = foreign_member.Decode(foreign_member.EXTERNAL / foreign_member.SCRIPT).loaded()
    reader = theirs._make_reader()  # noqa: SLF001
    series_root = raw / "train_series"
    records = RaptorFeed.records(raw / "train_series.csv")
    ours = RaptorFeed(series_root)

    for study in list(records)[: arguments.studies]:
        mine, my_mask = ours.study(study, records)
        their_volume, their_mask = theirs.build_study(study, records, str(series_root), reader)
        if not np.array_equal(mine, their_volume) or not np.array_equal(my_mask, their_mask):
            differing = int((mine != their_volume).any(axis=(1, 2)).sum())
            message = f"{study}: our decode differs from theirs on {differing} of {STACK_DEPTH} depths"
            raise ValueError(message)
        logger.info("%s: identical", study)
    logger.info("%d studies decode identically", arguments.studies)


if __name__ == "__main__":
    main()
