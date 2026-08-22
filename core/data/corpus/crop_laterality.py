"""Does the 130 mm crop cut the LATERAL compartment? `python -m core.data.corpus.crop_laterality`.

THE LAST CHEAP CANDIDATE knee-yrg left standing. Medial beats lateral by +0.0611 (meniscus) and +0.0437
(OA) corpus-wide, and three explanations are already refuted: the laterality flip (the gap survives where
tag and geometry agree), the slice band (`interpretations/kneemri/2026-08-17_slice_band_coverage.md` — a
symmetric cut cannot make an asymmetric deficit), and prevalence
(`interpretations/kneemri/2026-08-18_lateral_gap_is_not_prevalence.md` — matched counts move the gap by
0.0001). What remains is the IN-PLANE crop: a 130 mm box centred on the middle of the frame, with the
joint sitting off-centre, clips one compartment before the other, and the lateral one is the candidate.

WHY IT IS DECIDABLE WITHOUT A LABEL OR A CARD, which is the whole reason it is run now. The crop box and
the tissue are both in the array; which edge of the frame is LATERAL is in the header — the row direction
cosine's x component, with the knee's side. So the question becomes arithmetic: of the tissue this
acquisition imaged, what share does the crop discard on the lateral side, and what share on the medial?

WHY CORONAL AND AXIAL ONLY. Medial-lateral is IN-PLANE on those two and THROUGH-PLANE on a sagittal
series, where it is the band's axis and not the crop's — already measured, already refuted. A sagittal
slot is declined here rather than counted as symmetric, because it is silent, not balanced.

WHAT IT CANNOT SAY, and this bounds the claim in the same direction `crop_coverage` had to bound its own:
discarded TISSUE is not discarded COMPARTMENT. A crop that clips the lateral edge of the FRAME is clipping
thigh and skin as well as knee. So a crop that discards the SAME share on both sides cannot be producing an
asymmetric deficit, and that is the direction that kills the hypothesis cheaply; a crop that clips lateral
harder is a reason to rebuild a cache, not a result.
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from core.bootstrap import RESAMPLE_SEED, Bootstrap
from core.data.corpus.acquisition import Acquisition
from core.data.corpus.crop_centre import CROP_MM
from core.data.corpus.crop_coverage import UNCROPPED, CropCoverage
from core.data.corpus.slot_reader import SlotReader
from core.data.corpus.slots import Plane, Routing
from core.data.corpus.study import SlotPipeline, Study
from core.data.corpus.study_cache import CorpusSplit
from core.paths import DataRoot

logger = logging.getLogger(__name__)

IN_PLANE_AXES = (Plane.CORONAL, Plane.AXIAL)
"""The planes whose column axis IS the medial-lateral one. Sagittal carries it through-plane."""

MIN_AXIS_X = 0.5
"""How much of the column direction must lie along the patient's left-right axis to name a side.

An oblique acquisition whose columns run mostly superior-inferior has no lateral EDGE to speak of, and a
sign taken off a near-zero cosine is the sign of the rounding. Half a unit vector is 60 degrees of tilt —
past that the axis is a different axis, and the series is declined rather than guessed at.
"""

SLICES = 12
"""What the cache keeps per series, so the stack measured is the stack a member would read."""

SAMPLE = 400
"""Studies to decode, matching `crop_coverage` and `band_coverage` — the same sampling error, and the
same reason: the decode is the whole cost and a larger sample buys nothing this decision can use."""


@dataclass(frozen=True)
class SideClip:
    """What one series' crop discards on each side of the medial-lateral axis, as shares of its tissue."""

    lateral: float
    medial: float

    def asymmetry(self) -> float:
        """Lateral minus medial: positive means the crop takes more from the lateral side."""
        return self.lateral - self.medial


@dataclass(frozen=True)
class ClipRequest:
    """One study's measurement job, as the pool receives it."""

    identifier: str
    source: str
    planes: dict[str, str]
    slices: int = SLICES
    crop_mm: float = CROP_MM


class CropLaterality:
    """Which side of the knee the in-plane crop discards, per series and then over a sample."""

    def __init__(self, crop_mm: float = CROP_MM) -> None:
        self.crop_mm = crop_mm

    def lateral_column(self, acquisition: Acquisition, side: str | None) -> int | None:
        """+1 when the LAST column is lateral, -1 when the FIRST is, None when the series cannot say.

        Two facts multiply. The header's row direction cosine gives the patient's left per column step
        (+x is left in LPS). The knee's side turns left into lateral: on a LEFT knee the lateral
        compartment is the far-left one, so lateral is +x; on a RIGHT knee the midline lies to the
        patient's left of the joint and lateral is -x. An unresolved side leaves the mapping undefined,
        and an undefined mapping must not be defaulted — it would enter the mean with a coin-flip sign
        and pull the answer toward zero, which is the very verdict this measurement can return.
        """
        axis = acquisition.column_axis_x
        if axis is None or abs(axis) < MIN_AXIS_X or side not in ("L", "R"):
            return None
        towards_left = 1 if axis > 0 else -1
        return towards_left if side == "L" else -towards_left

    def clipped(self, stack: NDArray[np.float32], spacing: float | None, lateral: int) -> SideClip | None:
        """The tissue outside the crop on each side, as a share of the tissue in the whole acquisition.

        None where the crop does not apply — no spacing, or a frame already narrower than 130 mm, which
        is an acquisition the crop discards nothing from and therefore no evidence about which side it
        would discard from. Counting it as a perfect zero-zero would dilute an effect toward the verdict
        of no effect.

        The mask is `CropCoverage`'s, not a second threshold beside it: two coverage measurements that
        disagree about what tissue is are measuring their thresholds.
        """
        reader = SlotReader(crop_mm=self.crop_mm)
        taken = reader.window(stack, spacing, self.crop_mm)
        if taken is None:
            return None
        mask = CropCoverage().tissue(stack)
        total = float(mask.sum())
        if total == 0.0:
            return None
        columns = taken[1]
        before = float(mask[:, :, : columns.start].sum()) / total
        after = float(mask[:, :, columns.stop :].sum()) / total
        return SideClip(lateral=after, medial=before) if lateral > 0 else SideClip(lateral=before, medial=after)

    @staticmethod
    def measure_study(request: ClipRequest) -> list[SideClip]:
        """Every coronal or axial slot of one study, measured. A staticmethod so a pool can pickle it.

        A study that will not decode contributes NOTHING rather than a zero, for the same reason
        `crop_coverage` gives: an unreadable series is silent about the crop, and a zero would charge the
        crop for the DICOM library.
        """
        reader = SlotReader(crop_mm=UNCROPPED)
        pipeline = SlotPipeline(scheme=Routing().built(), reader=reader)
        planes = {slot.name: slot.plane for slot in pipeline.scheme.slots}
        measurement = CropLaterality(request.crop_mm)
        try:
            study = Study.from_directory(Path(request.source) / request.identifier, request.planes, pipeline)
            side = study.side()
            routed = pipeline.route(study.acquisitions, study.planes)
            read = [
                (
                    reader.cropped(acquisition, request.slices),
                    acquisition,
                    measurement.lateral_column(acquisition, side),
                )
                for name, acquisition in routed.items()
                if planes.get(name) in IN_PLANE_AXES
            ]
        except (OSError, ValueError, RuntimeError, IndexError) as failure:
            logger.info(f"  {request.identifier} did not decode: {type(failure).__name__}: {failure}")
            return []
        clips = [
            measurement.clipped(stack, acquisition.pixel_spacing, lateral)
            for stack, acquisition, lateral in read
            if stack is not None and lateral is not None
        ]
        return [clip for clip in clips if clip is not None]


@dataclass(frozen=True)
class CropLateralityReport:
    """The paired discard as prose — the rendering half of what `CropLaterality` used to be.

    Split for the reason the other reports were: measuring decodes four hundred studies uncropped, and
    rendering needs only the `SideClip` list they produced. Held together, the verdict line could not be
    exercised without a corpus to decode.
    """

    measured: list[SideClip]
    sampled: int
    corpus: int
    seed: int = RESAMPLE_SEED

    def lines(self) -> list[str]:
        """The two discards, their paired difference with an interval, and WHAT IT WAS MEASURED ON.

        Paired over the same series rather than compared as two populations: how much tissue a knee
        shows outside its joint varies far more between patients than between the two sides of one
        patient, and an unpaired interval would be that spread rather than this effect.

        The denominators lead rather than trail, because the mistake this project has already paid for
        three times is a share computed inside the subset that happened to parse, read as a share of the
        corpus.
        """
        measured = self.measured
        lines = [
            f"{len(measured)} coronal/axial series measured, "
            f"from {self.sampled} studies sampled of {self.corpus} in the corpus"
        ]
        if not measured:
            return lines
        deltas = np.array([clip.asymmetry() for clip in measured])
        low, high = Bootstrap.seeded(self.seed).mean_interval(deltas)
        clips = "the crop clips one side harder" if low > 0.0 or high < 0.0 else "SYMMETRIC — refuted as the asymmetry"
        lines.append(
            f"discarded lateral {np.mean([clip.lateral for clip in measured]):.4f}  "
            f"medial {np.mean([clip.medial for clip in measured]):.4f}  "
            f"lateral - medial {deltas.mean():+.4f} [{low:+.4f}, {high:+.4f}]  {clips}"
        )
        return lines


def main() -> None:
    """Decode a spread sample uncropped and price the crop's two edges against the knee's two sides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=SAMPLE, help="studies to decode")
    parser.add_argument("--slices", type=int, default=SLICES)
    parser.add_argument("--crop-mm", type=float, default=CROP_MM, help="the extent every member reads at")
    parser.add_argument("--workers", type=int, default=20, help="decode processes")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    split = CorpusSplit(DataRoot.from_config(arguments.paths).raw())
    identifiers = split.identifiers()
    planes = split.planes()
    # Every k-th rather than a random draw, matching `crop_coverage`: the sample is then a property of
    # the corpus order rather than of a seed, and repeating cannot quietly answer with other studies.
    stride = max(1, len(identifiers) // arguments.sample)
    sampled = identifiers[::stride][: arguments.sample]
    requests = [
        ClipRequest(identifier, str(split.source), planes, arguments.slices, arguments.crop_mm)
        for identifier in sampled
    ]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        measured = [clip for study in pool.map(CropLaterality.measure_study, requests) for clip in study]
    for line in CropLateralityReport(measured, len(sampled), len(identifiers)).lines():
        logger.info(line)


if __name__ == "__main__":
    main()
