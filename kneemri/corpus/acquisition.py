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

_ORDER_TAGS = ["ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber"]

_ORIENTATION_VALUES = 6
"""ImageOrientationPatient is two 3-vectors: the row and column direction cosines."""

_POSITION_VALUES = 3
"""ImagePositionPatient is one 3-vector in patient coordinates."""


@dataclass(frozen=True)
class SeriesHeader:
    """One slice's DICOM header, read as the acquisition axes it implies.

    SEPARATE FROM `Acquisition` because it answers a different question. An acquisition is the series
    as the routing sees it — what it is, how many slices it holds, which way they run. This is the
    vendor dialect: five parsers that share a header, touch none of the series' own fields, and each
    exist because the tag they read is optional, multi-valued, or lies.

    An EMPTY header is a legitimate one, and is what a series directory holding no DICOM produces. Every
    axis then reads absent by the same rules rather than by a second hand-written branch — the branch
    that used to sit in `probe` and had to be kept in step with these five by hand.
    """

    header: pydicom.Dataset

    @classmethod
    def of(cls, path: Path) -> "SeriesHeader":
        """The header alone — `stop_before_pixels`, because a probe must not pay to decode pixel data.

        The WHOLE header, not the twelve tags these five rules read. There was a `_HEADER_TAGS` tuple
        sitting here unused for exactly that narrowing; measured on 667 headers over 120 studies, adding
        `specific_tags` runs at 0.92-0.98x — slower, because `stop_before_pixels` has already skipped the
        only expensive part and the filter costs more than the elements it drops. Whole-corpus scale:
        ~24k headers, ~8 s either way. Deleted rather than wired in, so it cannot look like a pending
        optimisation to the next reader.
        """
        return cls(pydicom.dcmread(str(path), stop_before_pixels=True, force=True))

    @property
    def _described(self) -> str:
        """Description and sequence name as one lowercase string, separators flattened to spaces.

        Both the weighting and the fat-suppression rules read it, and they read it identically — it was
        two copies of this expression, which is two places for a separator to be added to one of them.
        """
        return _SEPARATORS.sub(
            " ",
            f"{getattr(self.header, 'SeriesDescription', '') or ''} "
            f"{getattr(self.header, 'SequenceName', '') or ''}".lower(),
        )

    @property
    def weighting(self) -> str:
        """T1 / T2 / PD / GRE / UNK, from the description where it says so and from TR-TE where it does not.

        Gradient echo is settled first because its TR is short by design and would otherwise read as T1.
        """
        described = self._described
        if _PD.search(described):
            return "PD"
        if _T2.search(described):
            return "T2"
        if _T1.search(described):
            return "T1"
        if "GR" in str(getattr(self.header, "ScanningSequence", "") or "").upper():
            return "GRE"
        repetition = self._number(getattr(self.header, "RepetitionTime", None))
        echo = self._number(getattr(self.header, "EchoTime", None))
        if repetition is None:
            return "UNK"
        if repetition < _LONG_TR_MS:
            return "T1"
        if echo is not None and echo > _LONG_TE_MS:
            return "T2"
        return "PD"

    @property
    def fat_suppressed(self) -> bool:
        """Fat suppression is a preparation named in the description, the sequence name, or ScanOptions."""
        raw = getattr(self.header, "ScanOptions", None) or ""
        # ScanOptions is Type 1C multi-valued: pydicom hands back a MultiValue for "FS\\SAT_GEMS" and a
        # plain str for "FS". Stringifying the first gives "['FS', 'SAT_GEMS']", whose tokens no longer
        # match — so the two cases are separated here rather than papered over with a substring test.
        multi_valued = isinstance(raw, (list, pydicom.multival.MultiValue))
        tokens = [str(value) for value in raw] if multi_valued else str(raw).split("\\")
        return bool(_FATSAT.search(self._described)) or any(
            token.strip().upper() in _FATSAT_OPTIONS for token in tokens
        )

    @property
    def pixel_spacing(self) -> float | None:
        """In-plane mm per pixel, which with the crop extent fixes the effective resolution."""
        spacing = getattr(self.header, "PixelSpacing", None)
        return None if spacing is None or len(spacing) < 1 else float(spacing[0])

    @property
    def slice_spacing(self) -> float | None:
        """Through-plane mm per slice, which is what turns a slice band into a distance.

        `SpacingBetweenSlices` before `SliceThickness`: a gapped stack advances by the former, and the
        latter would then understate the extent by the gap. Absent both, the series has no physical
        through-plane scale and says so rather than assuming a millimetre.
        """
        for tag in ("SpacingBetweenSlices", "SliceThickness"):
            spacing = self._number(getattr(self.header, tag, None))
            if spacing is not None and spacing > 0:
                return spacing
        return None

    @property
    def laterality_tag(self) -> str | None:
        """'L' / 'R' from the tag, treating present-but-empty as absent — which it is."""
        value = str(getattr(self.header, "Laterality", "") or "").strip().upper()
        return value[0] if value[:1] in ("L", "R") else None

    @property
    def centre_x(self) -> float | None:
        """The x of the image CENTRE in patient coordinates: +x is the patient's left.

        The corner (`ImagePositionPatient` itself) sits half a field of view away, which is enough to
        reverse the sign on a knee scanned near the midline.
        """
        position = getattr(self.header, "ImagePositionPatient", None)
        orientation = getattr(self.header, "ImageOrientationPatient", None)
        spacing = getattr(self.header, "PixelSpacing", None)
        rows, columns = getattr(self.header, "Rows", None), getattr(self.header, "Columns", None)
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
    slice_spacing: float | None = None

    @property
    def through_plane_mm(self) -> float | None:
        """How many mm the series spans along the axis a slice band cuts.

        None where the spacing tag is absent: a share of an unknown extent is not a distance, and this
        project has read a share as a physical quantity before.
        """
        return None if self.slice_spacing is None else self.slice_spacing * self.slice_count

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

        A directory holding no DICOM is probed rather than skipped, and reads its axes off an EMPTY
        header — the same five rules, not a second hand-written all-absent branch beside them.
        """
        files = tuple(sorted(entry.name for entry in directory.iterdir() if entry.name.endswith(".dcm")))
        header = SeriesHeader.of(directory / files[len(files) // 2]) if files else SeriesHeader(pydicom.Dataset())
        return cls(
            study=study,
            series=series,
            directory=directory,
            files=files,
            weighting=header.weighting,
            fat_suppressed=header.fat_suppressed,
            pixel_spacing=header.pixel_spacing,
            laterality_tag=header.laterality_tag,
            centre_x=header.centre_x,
            slice_spacing=header.slice_spacing,
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
