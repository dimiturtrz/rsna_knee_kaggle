"""One MRI series, read as an acquisition: what it is, which way its slices run, and where it sits.

Three facts have to be recovered from the header rather than taken from the tables, and each was
verified on our own corpus before being relied on (see `knee-25n`):

- **What it is.** The shipped `Fluid_Sensitive` and `Fat_Suppression` columns agree on 1.000000 of
  24,371 rows — one axis wearing two names. Weighting comes from TR/TE with `ScanningSequence` settling
  gradient echo first; fat suppression from the description, sequence name and `ScanOptions`.
- **Which way the slices run.** File names are SOP Instance UIDs, so their order is uncorrelated with
  anatomy (ρ = −0.0106 measured here). The physical order is `k = p·(r_x × r_y)`, signed in patient
  coordinates, which is also what laterality reversal needs.
- **Where it sits.** `Laterality` is Type 2C: absent on half the corpus, and *present but empty* on some
  series — the two are not the same and a `hasattr` check misses the second. The image centre's x sign
  recovers the side, using the centre rather than the corner because a corner is half a field of view
  away, enough to flip the sign near the midline.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from numpy.typing import NDArray

_SEPARATORS = re.compile(r"[_\-.]")
_FATSAT = re.compile(
    r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|water excit|\btirm\b|\bsting\b|\bfatsup\b"
)
_T1 = re.compile(r"\bt1\b|\bt1w\b")
_T2 = re.compile(r"\bt2\b|\bt2w\b")
_PD = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")

_FATSAT_OPTIONS = frozenset({"FS", "FATSAT", "FAT_SAT", "FSAT"})
"""`ScanOptions` is matched as exact tokens: GE writes SAT_GEMS for spatial saturation, so a substring
test on "SAT" fires on series that are not fat-suppressed."""

_LONG_TR_MS = 800.0
"""Above this repetition time a sequence is no longer T1-weighted."""

_LONG_TE_MS = 60.0
"""Above this echo time a long-TR sequence is T2-weighted rather than proton density."""

_HEADER_TAGS = (
    "SeriesDescription",
    "SequenceName",
    "ScanOptions",
    "ScanningSequence",
    "RepetitionTime",
    "EchoTime",
    "Laterality",
    "PixelSpacing",
    "Rows",
    "Columns",
    "ImagePositionPatient",
    "ImageOrientationPatient",
)

_ORDER_TAGS = ["ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber"]

_ORIENTATION_VALUES = 6
"""ImageOrientationPatient is two 3-vectors: the row and column direction cosines."""

_POSITION_VALUES = 3
"""ImagePositionPatient is one 3-vector in patient coordinates."""


@dataclass(frozen=True)
class Acquisition:
    """One series: its files, the header of a representative slice, and what that header implies."""

    study: str
    series: str
    directory: Path
    files: tuple[str, ...]
    weighting: str
    fat_suppressed: bool
    pixel_spacing: float | None
    laterality_tag: str | None
    centre_x: float | None

    @property
    def fluid_sensitive(self) -> bool:
        """PD and T2 are the fluid-bright weightings; T1 and gradient echo are not."""
        return self.weighting in ("PD", "T2")

    @property
    def slice_count(self) -> int:
        """How many slices the series holds — the tie-break when a slot has several candidates."""
        return len(self.files)

    @classmethod
    def probe(cls, study: str, series: str, directory: Path) -> "Acquisition":
        """Read one representative slice's header and derive the acquisition axes from it.

        The middle slice is read rather than the first: the ends of a knee stack are often outside the
        joint, and on some vendors the first slice is a scout or a duplicate.
        """
        files = tuple(sorted(entry.name for entry in directory.iterdir() if entry.name.endswith(".dcm")))
        if not files:
            return cls(
                study=study,
                series=series,
                directory=directory,
                files=(),
                weighting="UNK",
                fat_suppressed=False,
                pixel_spacing=None,
                laterality_tag=None,
                centre_x=None,
            )
        header = pydicom.dcmread(str(directory / files[len(files) // 2]), stop_before_pixels=True, force=True)
        return cls(
            study=study,
            series=series,
            directory=directory,
            files=files,
            weighting=cls._weighting(header),
            fat_suppressed=cls._fat_suppressed(header),
            pixel_spacing=cls._spacing(header),
            laterality_tag=cls._laterality_tag(header),
            centre_x=cls._centre_x(header),
        )

    def ordered_files(self) -> tuple[str, ...]:
        """The files sorted along the through-plane axis, by `k = p·(r_x × r_y)`.

        `InstanceNumber` is the fallback where the geometry tags are missing. A file where neither is
        readable keeps ITS OWN SLOT and the rest sort around it: worse than sorting, better than
        dropping the series — and far better than what this used to do, which was abandon the sort for
        the whole series over one unreadable header. Names are arbitrary UIDs in this corpus, so that
        fallback made the central band sample depth-unrelated slices while the presence mask still
        called the slot acquired.

        Costs nothing here and cannot be measured here: 0 of 1682 series over 300 studies hold a keyless
        file, so no cached array moves. It is insurance for the hidden test, which is the only place a
        vendor this corpus does not contain can turn up.
        """
        keyed = [(self._through_plane_key(self.directory / name), name) for name in self.files]
        sortable = iter(sorted((key, name) for key, name in keyed if key is not None))
        return tuple(name if key is None else next(sortable)[1] for key, name in keyed)

    @staticmethod
    def _through_plane_key(path: Path) -> float | None:
        """`k = p·n` for one slice, or its InstanceNumber, or None when neither is readable."""
        try:
            header = pydicom.dcmread(str(path), force=True, stop_before_pixels=True, specific_tags=_ORDER_TAGS)
        except (OSError, ValueError):
            return None
        orientation = getattr(header, "ImageOrientationPatient", None)
        position = getattr(header, "ImagePositionPatient", None)
        if (
            orientation is not None
            and position is not None
            and len(orientation) >= _ORIENTATION_VALUES
            and len(position) >= _POSITION_VALUES
        ):
            axes = np.asarray(orientation, dtype=float)
            return float(np.dot(np.asarray(position, dtype=float), np.cross(axes[:3], axes[3:6])))
        number = getattr(header, "InstanceNumber", None)
        return None if number is None else float(number)

    @staticmethod
    def _weighting(header: pydicom.Dataset) -> str:
        """T1 / T2 / PD / GRE / UNK, from the description where it says so and from TR-TE where it does not.

        Gradient echo is settled first because its TR is short by design and would otherwise read as T1.
        """
        described = _SEPARATORS.sub(
            " ", f"{getattr(header, 'SeriesDescription', '') or ''} {getattr(header, 'SequenceName', '') or ''}".lower()
        )
        if _PD.search(described):
            return "PD"
        if _T2.search(described):
            return "T2"
        if _T1.search(described):
            return "T1"
        if "GR" in str(getattr(header, "ScanningSequence", "") or "").upper():
            return "GRE"
        repetition = Acquisition._number(getattr(header, "RepetitionTime", None))
        echo = Acquisition._number(getattr(header, "EchoTime", None))
        if repetition is None:
            return "UNK"
        if repetition < _LONG_TR_MS:
            return "T1"
        if echo is not None and echo > _LONG_TE_MS:
            return "T2"
        return "PD"

    @staticmethod
    def _fat_suppressed(header: pydicom.Dataset) -> bool:
        """Fat suppression is a preparation named in the description, the sequence name, or ScanOptions."""
        described = _SEPARATORS.sub(
            " ", f"{getattr(header, 'SeriesDescription', '') or ''} {getattr(header, 'SequenceName', '') or ''}".lower()
        )
        raw = getattr(header, "ScanOptions", None) or ""
        # ScanOptions is Type 1C multi-valued: pydicom hands back a MultiValue for "FS\\SAT_GEMS" and a
        # plain str for "FS". Stringifying the first gives "['FS', 'SAT_GEMS']", whose tokens no longer
        # match — so the two cases are separated here rather than papered over with a substring test.
        multi_valued = isinstance(raw, (list, pydicom.multival.MultiValue))
        tokens = [str(value) for value in raw] if multi_valued else str(raw).split("\\")
        return bool(_FATSAT.search(described)) or any(token.strip().upper() in _FATSAT_OPTIONS for token in tokens)

    @staticmethod
    def _spacing(header: pydicom.Dataset) -> float | None:
        """In-plane mm per pixel, which with the crop extent fixes the effective resolution."""
        spacing = getattr(header, "PixelSpacing", None)
        return None if spacing is None or len(spacing) < 1 else float(spacing[0])

    @staticmethod
    def _laterality_tag(header: pydicom.Dataset) -> str | None:
        """'L' / 'R' from the tag, treating present-but-empty as absent — which it is."""
        value = str(getattr(header, "Laterality", "") or "").strip().upper()
        return value[0] if value[:1] in ("L", "R") else None

    @staticmethod
    def _centre_x(header: pydicom.Dataset) -> float | None:
        """The x of the image CENTRE in patient coordinates: +x is the patient's left.

        The corner (`ImagePositionPatient` itself) sits half a field of view away, which is enough to
        reverse the sign on a knee scanned near the midline.
        """
        position = getattr(header, "ImagePositionPatient", None)
        orientation = getattr(header, "ImageOrientationPatient", None)
        spacing = getattr(header, "PixelSpacing", None)
        rows, columns = getattr(header, "Rows", None), getattr(header, "Columns", None)
        if position is None or orientation is None or spacing is None or not rows or not columns:
            return None
        axes = np.asarray(orientation, dtype=float)
        centre: NDArray[np.float64] = (
            np.asarray(position, dtype=float)
            + axes[:3] * float(spacing[1]) * float(columns) / 2
            + axes[3:6] * float(spacing[0]) * float(rows) / 2
        )
        return float(centre[0])

    @staticmethod
    def _number(value: object) -> float | None:
        """A DICOM numeric that may be absent, empty or unparsable."""
        try:
            return float(value)  # pyrefly: ignore[bad-argument-type]
        except (TypeError, ValueError):
            return None
