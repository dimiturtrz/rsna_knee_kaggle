"""Do a study's slots agree WHERE the knee is: `python -m core.data.corpus.crop_alignment`.

THE QUESTION, AND WHY IT IS NOT THE ONE `crop_coverage` ANSWERS. `crop_coverage` scores how much tissue
each crop centre KEEPS, per series, and knee-yrf was closed on that statistic: tissue-centring retains
+0.0294 more at 130 mm, which is real, knob-sized, and did not pay for a 99 GB cache rebuild. Retention
is a WITHIN-SERIES quantity. It cannot say whether two slots of the same study place the knee at the same
spot inside their own crops, and that is a different fact with a different consequence.

WHY IT MATTERS NOW. CoPAS (Nature Communications 15:7637) measures its single largest ablation component
at +0.056 mean AUC for cropping to a segmented ROI, and the mechanism the authors credit is not
retention: "cropping along ROI can align the spatial position across sequences". Every slot-aggregation
arm this project has run has landed inside the noise floor. If our image-centred box places the knee at a
different offset in each slot, then every one of those arms was fusing views that do not correspond, and
the aggregator was never the thing at fault.

WHAT IS MEASURED. Per slot, the offset from the image centre to the tissue centroid, in millimetres — the
displacement the current default crop does not correct. Then, per study, the SPREAD of that offset's
MAGNITUDE across the study's routed slots.

MAGNITUDE, AND ONLY MAGNITUDE, ACROSS PLANES. A sagittal slice's rows run superior-inferior while an
axial slice's run anterior-posterior, so comparing a sagittal dy against an axial dy compares two
different anatomical axes and means nothing. The magnitude is rotation-invariant and survives the
comparison; the signed components are reported PER PLANE, where the axes are consistent and the reading
is clean.

WHAT IT CANNOT SAY. This is a proxy for misregistration, not a registration error — there is no
ground-truth joint centre here, and an Otsu mass centroid is dragged by coil, fat and the far limb
(knee-yrf measured that: tissue-centring's worst series is worse than image-centring's at 80 mm). It
bounds the question from one side: if the spread is small, the slots already agree and the alignment
mechanism is not available here at any price.
"""

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from core import cli
from core.data.corpus.audit import UNCROPPED, AuditRequest, CorpusAudit, Reading
from core.data.corpus.crop_centre import ImageCentre, TissueCentre
from core.data.corpus.crop_coverage import SLICES
from core.data.corpus.study_cache import CorpusSample

logger = logging.getLogger(__name__)

AUDIT = CorpusAudit(reading=Reading(crop_mm=UNCROPPED), unit="studies", slices=SLICES)
"""Read UNCROPPED, because the displacement measured is the one the crop WOULD apply.

Taken from an already-cropped stack it would measure what is left after the error rather than the error.
"""

MIN_SLOTS_FOR_SPREAD = 2
"""A study with one routed slot has nothing to disagree with, so it abstains rather than scoring zero."""

PERCENTILES = (50, 90)
"""The median and the tail. The tail is the half that decides this: a crop error that only bites one study
in ten is exactly the shape of a defect that hides inside a mean AUC."""


@dataclass(frozen=True)
class SlotOffset:
    """Where one slot's tissue sits relative to the box the default crop takes about the image centre."""

    plane: str
    down_mm: float
    across_mm: float

    def magnitude(self) -> float:
        """How far the knee is from the point the crop assumes it occupies, in millimetres.

        Rotation-invariant, which is what lets a sagittal slot be compared against an axial one at all:
        their row axes name different anatomical directions, and only the distance survives that.
        """
        return float(np.hypot(self.down_mm, self.across_mm))


@dataclass(frozen=True)
class StudyOffsets:
    """One study's routed slots, still carrying the identifier that names it.

    The identifier travels with the measurement because the aggregate report is not the only consumer:
    a per-study spread can only be joined against a per-study holdout error if both are keyed, and an
    aggregate that has already dropped the key cannot be re-keyed afterwards.
    """

    identifier: str
    offsets: list[SlotOffset]


@dataclass(frozen=True)
class CropAlignment:
    """Whether the routed slots of one study place the knee at the same offset inside their own frames."""

    def offset(self, stack: NDArray[np.float32], spacing: float | None, plane: str) -> SlotOffset | None:
        """One slot's tissue centroid minus the image centre, in millimetres.

        None without a pixel spacing rather than in pixels: a displacement in pixels is not comparable
        between a 256-matrix and a 384-matrix acquisition, and this whole measurement is a comparison
        BETWEEN acquisitions.
        """
        if spacing is None:
            return None
        down, across = TissueCentre().of(stack)
        middle_down, middle_across = ImageCentre().of(stack)
        return SlotOffset(plane, (down - middle_down) * spacing, (across - middle_across) * spacing)

    def spread(self, offsets: list[SlotOffset]) -> float | None:
        """The mean pairwise gap in offset MAGNITUDE across a study's slots, in millimetres.

        Mean pairwise rather than max-minus-min, so one dragged centroid cannot define the study — the
        Otsu mass centre is known to be pulled by coil and far limb, and a range statistic would report
        that failure as misalignment.
        """
        if len(offsets) < MIN_SLOTS_FOR_SPREAD:
            return None
        magnitudes = [offset.magnitude() for offset in offsets]
        gaps = [abs(first - second) for first, second in combinations(magnitudes, 2)]
        return float(np.mean(gaps))

    def report(self, measured: list[list[SlotOffset]], sample: CorpusSample) -> list[str]:
        """The per-plane offset distribution, then the within-study spread, then the denominators.

        The denominators are stated because a share computed over the studies that happened to decode is
        not a share over the corpus, and this project has read one as the other before.
        """
        slots = [offset for study in measured for offset in study]
        spreads = [found for found in (self.spread(study) for study in measured) if found is not None]
        lines = [
            f"{sample.line(len(measured), 'studies')}; "
            f"{len(slots)} routed slots, {len(spreads)} studies with at least {MIN_SLOTS_FOR_SPREAD} slots",
            f"{'plane':14s} {'|offset| p50':>13s} {'p90':>8s} {'down p50':>10s} {'across p50':>12s} {'slots':>7s}",
        ]
        lines.extend(self.plane_lines(slots))
        if spreads:
            low, high = np.percentile(spreads, PERCENTILES)
            lines.append(
                f"within-study spread of |offset| across slots: p50 {low:.1f} mm, p90 {high:.1f} mm, "
                f"max {max(spreads):.1f} mm"
            )
        return lines

    def plane_lines(self, slots: list[SlotOffset]) -> list[str]:
        """One row per plane the slots cover, planes in name order."""
        return [
            self.plane_line(plane, [offset for offset in slots if offset.plane == plane])
            for plane in sorted({offset.plane for offset in slots})
        ]

    def plane_line(self, plane: str, offsets: list[SlotOffset]) -> str:
        """One plane's row: magnitude median and tail, plus the SIGNED components its axes make readable."""
        magnitudes = np.array([offset.magnitude() for offset in offsets])
        median, tail = np.percentile(magnitudes, PERCENTILES)
        down = np.median([offset.down_mm for offset in offsets])
        across = np.median([offset.across_mm for offset in offsets])
        return f"{plane:14s} {median:13.1f} {tail:8.1f} {down:10.1f} {across:12.1f} {magnitudes.size:7d}"

    def dump(self, destination: Path, decoded: list["StudyOffsets"]) -> None:
        """Write each study's spread beside its identifier, abstentions included as nan rather than dropped.

        A study with one routed slot has no spread, and writing it as nan keeps it visible to the consumer
        as an abstention. Dropping it would let a join silently narrow to the multi-slot studies while
        still reporting a count taken over all of them.
        """
        spreads = [self.spread(study.offsets) for study in decoded]
        np.savez(
            destination,
            studies=np.array([study.identifier for study in decoded], dtype="<U64"),
            spread=np.array([np.nan if found is None else found for found in spreads], dtype=np.float32),
            slots=np.array([len(study.offsets) for study in decoded], dtype=np.int32),
        )

    @staticmethod
    def measure_study(request: AuditRequest) -> list["StudyOffsets"]:
        """Every routed slot of one study, offset — STILL NAMED where none of them placed.

        A LIST of exactly one study, because the pool flattens and a bare object cannot be extended into
        the result. A study that placed nothing stays in it rather than vanishing: `dump` writes an
        abstention as nan precisely so a join can tell a study that did not measure from one that was
        never drawn, and a worker that dropped it would take that distinction with it.
        """
        alignment = CropAlignment()
        found = [
            alignment.offset(stack, acquisition.pixel_spacing, slot)
            for slot, acquisition, stack in request.routed().decoded(request.slices)
            if stack is not None
        ]
        return [StudyOffsets(request.identifier, [offset for offset in found if offset is not None])]


def main() -> None:
    """Decode a spread sample of studies uncropped and report how far each slot's knee sits off centre."""
    command = cli.Command(__doc__, logger)
    AUDIT.flags(command)
    parser = command.parser
    parser.add_argument("--slices", type=int, default=SLICES)
    parser.add_argument(
        "--studies",
        type=Path,
        default=None,
        help="a file of identifiers, one per line, to measure INSTEAD of a spread sample of the corpus. "
        "The spread sample answers what the corpus looks like; a named list is what lets the spread be "
        "joined against a holdout record, which needs the same studies on both sides, not a sample of them",
    )
    parser.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="write the per-study spread to this .npz, keyed by identifier, alongside the report",
    )
    arguments = command.parsed()

    audit = AUDIT.with_slices(arguments.slices)
    alignment = CropAlignment()
    sample, requests = audit.drawn(command, arguments, studies=arguments.studies)
    decoded = audit.mapped(CropAlignment.measure_study, requests, arguments.workers)
    # The report is over the studies that PLACED; the dump keeps the abstentions, which is what a join
    # needs to tell a study that did not measure from one that was never drawn.
    placed = [study for study in decoded if study.offsets]
    audit.reported(command, alignment.report([study.offsets for study in placed], sample), sample, len(placed))
    if arguments.dump:
        alignment.dump(arguments.dump, decoded)
        logger.info(f"per-study spread written to {arguments.dump}")


if __name__ == "__main__":
    main()
