"""Do a study's slots image the same anatomy in patient space: `python -m core.data.corpus.slot_geometry`.

THE HALF `crop_alignment` COULD NOT SEE. That tool measured, per study, how far each slot's tissue sits
from its own image centre, and answered decisively: 5.7 mm p50 between slots, under half an encoder
feature cell. But it is an IN-PLANE statistic. It compares row/column offsets inside each slot's own
frame and never leaves that frame, so it cannot say whether two acquisitions are looking at the same
part of the knee at all. Two series can each be perfectly centred on their own tissue and still cover
different volumes.

WHAT IS MEASURED, AND WHY IT IS HEADER-ONLY. Every slice carries `ImagePositionPatient` (the patient-space
corner of the slice) and `ImageOrientationPatient` (the row and column direction cosines). Together with
`Rows`, `Columns` and the pixel spacing they place the slice's CENTRE at a point in the shared patient
frame, and their cross product gives the slice normal. So for each routed slot: a 3-vector centre, a unit
normal, and the extent the stack spans along that normal. No pixel is decoded, which is why this runs over
the corpus in seconds rather than the twenty minutes `crop_alignment` needed.

THE TWO NUMBERS THAT DECIDE knee-vigl. Between every pair of slots in a study: the ANGLE between their
normals, and the DISTANCE between their volume centres in millimetres. A reformat that rotates one slot
into another's plane is only meaningful if the two cover the same anatomy — the angle says whether the
planes are the orthogonal set the method assumes, and the distance says how far a reformatted volume would
have to be translated before the two could be stacked. A large centre distance means our slots are
genuinely unregistered in three dimensions, and the reformat has to carry a shift it currently does not.
A small one means they already share a volume and only the sampling direction differs.

WHAT IT CANNOT SAY. The centre is the middle slice's geometric centre, not a joint landmark: it inherits
whatever the technologist framed, and a series that ran long in one direction moves its centre without
moving the knee. It bounds the question rather than answering it — a small spread here rules the axis out,
a large one localises it but still needs the anatomy checked before a shift is built.
"""

import logging
from dataclasses import dataclass
from itertools import combinations, permutations

import numpy as np
import pydicom
from numpy.typing import NDArray

from core import cli
from core.data.corpus.acquisition import Acquisition
from core.data.corpus.audit import UNCROPPED, AuditRequest, CorpusAudit, Reading
from core.data.corpus.study_cache import CorpusSample

logger = logging.getLogger(__name__)

AUDIT = CorpusAudit(reading=Reading(crop_mm=UNCROPPED), unit="studies with geometry", workers=8)
"""Headers only — no pixel is decoded, so the crop never applies and eight processes are plenty.

The draw is the harness's every-k-th 400, which is what makes this the same studies `crop_coverage` and
`crop_alignment` measured: the three readings compose only under one sampler. It was spelled by hand
here, under a comment claiming exactly that — which is the drift `CorpusSample` was written to end.
"""

GEOMETRY_TAGS = ["ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns"]
"""The five tags that place a slice in the patient frame. Read with `specific_tags` because this is the
one measurement in the corpus tools that is header-bound rather than decode-bound."""

ORIENTATION_VALUES = 6
POSITION_VALUES = 3
HALF = 2.0
RIGHT_ANGLE = 90.0
PERCENTILES = (50, 90)
ORTHOGONAL_TOLERANCE_DEG = 20.0
"""How far from 90 degrees two normals may sit and still be called an orthogonal pair. Wide, because the
knee is not scanned to the scanner axes — sagittal series are routinely angled to the femoral condyles."""

DISTANCE_LEGS = ("centre distance", "  through-plane leg", "  in-plane leg")
"""The three millimetre columns of a `SlotGeometry.pairs` row after the angle, in that order."""


@dataclass(frozen=True)
class SlotFrame:
    """One slot's place in the shared patient coordinate frame: where it sits and which way it faces."""

    plane: str
    centre: tuple[float, float, float]
    normal: tuple[float, float, float]
    span_mm: float | None
    """How far the stack reaches along its own normal — the DENOMINATOR the through-plane leg needs. A
    27 mm displacement is a defect inside a 40 mm stack and a rounding error inside a 140 mm one, and this
    project has read a distance without its denominator before."""

    def covers(self, other: "SlotFrame") -> bool | None:
        """Whether the other slot's centre falls inside the anatomy this slot actually imaged.

        None where the span is unknown, because a coverage claim over an unknown extent is not a claim.
        """
        if self.span_mm is None:
            return None
        return self.through_plane_to(other) <= self.span_mm / HALF

    def angle_to(self, other: "SlotFrame") -> float:
        """Degrees between two slots' normals, folded into [0, 90].

        Folded because a normal's SIGN is a scan-direction convention, not a geometric fact: a coronal
        series acquired posterior-to-anterior has the opposite normal to one acquired the other way, and
        reading that as 180 degrees of disagreement would report every such pair as maximally misaligned.
        """
        first, second = np.asarray(self.normal), np.asarray(other.normal)
        cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    def offset_to(self, other: "SlotFrame") -> float:
        """Millimetres between two slots' volume centres in the patient frame."""
        return float(np.linalg.norm(np.asarray(self.centre) - np.asarray(other.centre)))

    def through_plane_to(self, other: "SlotFrame") -> float:
        """How far the other slot's centre sits OUT of this slot's mid-slice, along this slot's normal.

        The half of the displacement `crop_alignment` was structurally blind to: an in-plane statistic
        measures inside the slice, and this is the component perpendicular to it. Taken with this slot's
        own normal rather than a shared axis, because it answers a per-slot question — how far through its
        own stack would this slot have to travel to reach what the other one is centred on.
        """
        gap = np.asarray(other.centre) - np.asarray(self.centre)
        return float(abs(np.dot(gap, np.asarray(self.normal))))

    def in_plane_to(self, other: "SlotFrame") -> float:
        """The remainder of the displacement, lying inside this slot's own slice plane.

        Reported beside the through-plane component so the total cannot be credited to the wrong axis:
        these two are the legs of a right triangle whose hypotenuse is `offset_to`.
        """
        total, through = self.offset_to(other), self.through_plane_to(other)
        return float(np.sqrt(max(total * total - through * through, 0.0)))


@dataclass(frozen=True)
class SlotGeometry:
    """Whether the routed slots of one study are orthogonal views of one volume or unrelated framings."""

    def frame(self, acquisition: Acquisition, plane: str) -> SlotFrame | None:
        """The middle slice's patient-space centre and normal, or None where the geometry tags are absent.

        The MIDDLE slice for the same reason `Acquisition.probe` reads it: the ends of a knee stack are
        often outside the joint, so a first-slice corner would place the volume where the scan started
        rather than where the knee is.
        """
        if not acquisition.files:
            return None
        name = acquisition.files[len(acquisition.files) // 2]
        try:
            header = pydicom.dcmread(
                str(acquisition.directory / name), force=True, stop_before_pixels=True, specific_tags=GEOMETRY_TAGS
            )
        except (OSError, ValueError):
            return None
        return self.placed(header, plane, acquisition.through_plane_mm)

    def placed(self, header: pydicom.Dataset, plane: str, span_mm: float | None = None) -> SlotFrame | None:
        """A slice header read as a centre point and a normal, both in patient millimetres.

        The corner is walked to the centre along the slice's OWN axes — `position + (rows/2) * spacing *
        row_cosine + (columns/2) * spacing * column_cosine` — because a matrix centre expressed in the
        patient frame is what makes two differently-oriented series comparable at all.
        """
        orientation = getattr(header, "ImageOrientationPatient", None)
        position = getattr(header, "ImagePositionPatient", None)
        spacing = getattr(header, "PixelSpacing", None)
        rows, columns = getattr(header, "Rows", None), getattr(header, "Columns", None)
        if orientation is None or position is None or spacing is None or rows is None or columns is None:
            return None
        if len(orientation) < ORIENTATION_VALUES or len(position) < POSITION_VALUES:
            return None
        axes = np.asarray(orientation, dtype=float)
        row_axis, column_axis = axes[:POSITION_VALUES], axes[POSITION_VALUES:ORIENTATION_VALUES]
        corner = np.asarray(position, dtype=float)
        row_mm, column_mm = float(spacing[0]), float(spacing[-1])
        centre = corner + row_axis * (float(columns) / HALF) * column_mm + column_axis * (float(rows) / HALF) * row_mm
        return SlotFrame(plane, self.triple(centre), self.triple(np.cross(row_axis, column_axis)), span_mm)

    def triple(self, vector: NDArray[np.float64]) -> tuple[float, float, float]:
        """A 3-vector as a plain tuple, so a frame stays hashable and picklable across the pool."""
        return float(vector[0]), float(vector[1]), float(vector[2])

    def pairs(self, frames: list[SlotFrame]) -> list[tuple[float, float, float, float]]:
        """Every within-study slot pair as `(angle, centre distance, through-plane leg, in-plane leg)`, in
        degrees and millimetres.

        The legs are taken with the FIRST slot's normal, and each unordered pair is listed once: which
        slot is asked changes the split but not the total, and reporting both directions would double-count
        the same displacement.
        """
        return [
            (
                first.angle_to(second),
                first.offset_to(second),
                first.through_plane_to(second),
                first.in_plane_to(second),
            )
            for first, second in combinations(frames, 2)
        ]

    def coverage(self, frames: list[SlotFrame]) -> list[bool]:
        """For every ORDERED slot pair, whether the first slot imaged what the second is centred on.

        Ordered, unlike `pairs`, because coverage is not symmetric: a 40-slice sagittal stack can easily
        contain the centre of a 12-slice axial one while the axial stack contains nothing of the sagittal.
        """
        return [covered for first, second in permutations(frames, 2) if (covered := first.covers(second)) is not None]

    def report(self, measured: list[list[SlotFrame]], sample: CorpusSample) -> list[str]:
        """The pair distributions, then how many pairs are the orthogonal set a reformat assumes.

        The denominators lead because a share over the studies whose headers carried geometry is not a
        share over the corpus, and this project has read one as the other before. Through
        `CorpusSample.line` rather than spelled here, so the sentence cannot drift from the draw that
        produced it — which is the other half of the drift this module carried.
        """
        frames = [frame for study in measured for frame in study]
        pairs = [pair for study in measured for pair in self.pairs(study)]
        lines = [
            f"{sample.line(len(measured), 'studies with geometry')}; "
            f"{len(frames)} slots with geometry, {len(pairs)} within-study slot pairs"
        ]
        if not pairs:
            return lines
        covered = [seen for study in measured for seen in self.coverage(study)]
        return lines + self._distributions(pairs) + self._covered(covered)

    @staticmethod
    def _distributions(pairs: list[tuple[float, float, float, float]]) -> list[str]:
        """The angle and the three distance legs as percentiles, then the share of pairs that are orthogonal."""
        columns = np.array(pairs).T
        angle_low, angle_high = np.percentile(columns[0], PERCENTILES)
        orthogonal = int((np.abs(columns[0] - RIGHT_ANGLE) <= ORTHOGONAL_TOLERANCE_DEG).sum())
        legs = [
            f"{name:22s} between slots: p50 {low:.1f} mm, p90 {high:.1f} mm"
            for name, (low, high) in zip(
                DISTANCE_LEGS, (np.percentile(column, PERCENTILES) for column in columns[1:]), strict=True
            )
        ]
        return [
            f"normal angle between slots:   p50 {angle_low:.1f} deg, p90 {angle_high:.1f} deg",
            *legs,
            f"pairs within {ORTHOGONAL_TOLERANCE_DEG:.0f} deg of orthogonal: {orthogonal}/{len(pairs)} "
            f"({100 * orthogonal / len(pairs):.1f}%)",
        ]

    @staticmethod
    def _covered(covered: list[bool]) -> list[str]:
        """The coverage share over ordered pairs, or nothing when no slot's span was known."""
        if not covered:
            return []
        inside = sum(covered)
        return [
            f"ordered pairs where the first slot IMAGED the second's centre: {inside}/{len(covered)} "
            f"({100 * inside / len(covered):.1f}%)"
        ]

    @staticmethod
    def measure_study(request: AuditRequest) -> list[list[SlotFrame]]:
        """Every routed slot of one study, placed — as ONE group, because the pairs are within-study.

        A list of at most one study rather than a flat list of frames: the pool flattens what a worker
        returns, and a frame that has lost which study it came from cannot be paired with anything.
        """
        geometry = SlotGeometry()
        routed = request.routed()
        placed = [geometry.frame(acquisition, name) for name, acquisition in routed.slots.items()]
        frames = [frame for frame in placed if frame is not None]
        return [frames] if frames else []


def main() -> None:
    """Place every routed slot of a sample of studies in patient space and report the pairwise geometry."""
    command = cli.Command(__doc__, logger)
    AUDIT.flags(command)
    arguments = command.parsed()

    sample, requests = AUDIT.drawn(command, arguments)
    measured = AUDIT.mapped(SlotGeometry.measure_study, requests, arguments.workers)
    AUDIT.reported(command, SlotGeometry().report(measured, sample), sample, len(measured))


if __name__ == "__main__":
    main()
