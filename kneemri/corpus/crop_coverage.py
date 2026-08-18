"""Which crop centre actually keeps the knee: `python -m kneemri.corpus.crop_coverage`.

THE QUESTION knee-yrf was opened on and has never answered: `TissueCentre` exists, is plumbed through
six places and is opt-in, and NOTHING has been trained under it. Which centre is better is unmeasured,
and the bead's own evidence is an offset distribution over 60 studies from a scratchpad script that was
never committed.

WHY THIS AND NOT AN AUC. There is no ground-truth joint centre in this corpus, so "closer to the truth"
is not decidable, and the obvious alternative — train under each centre and compare — costs a GPU that
is not free and answers only after the fact. What IS decidable without a label is how much of the imaged
tissue each centre's crop box keeps. That is the mechanism the strategy exists for: a centre is only
worth anything if the box it picks contains more knee.

WHAT THE ANSWER GATES. At 130 mm the margin is about 15 mm and the two centres are expected to be near
indistinguishable — that is the prediction, and a large gap there would instead say the image centre is
worse than the offset distribution implies. At the 80 mm extent knee-1nn's multi-scale reading is gated
on, a half-extent of 40 mm has no margin for an 18 mm error, so this is where a difference must appear if
the strategy is worth a cache rebuild. If tissue-centring does NOT keep more at 80 mm, training under it
is wasted GPU and knee-1nn should read tight crops about the image centre after all.

WHAT IT CANNOT SAY. Retained tissue is not retained JOINT: a crop that drifts toward the calf keeps
tissue while losing the knee. This bounds the failure from one side only — a centre that keeps LESS
tissue at 80 mm cannot be the one keeping more joint, and that is the direction the decision needs.
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from core.bootstrap import RESAMPLE_SEED, Bootstrap
from core.paths import DataRoot
from kneemri.corpus.crop_centre import CENTRES, CROP_MM, TissueCentre
from kneemri.corpus.slot_reader import SlotReader
from kneemri.corpus.slots import Routing
from kneemri.corpus.study import SlotPipeline, Study
from kneemri.corpus.study_cache import CorpusSplit

logger = logging.getLogger(__name__)

UNCROPPED = 1.0e9
"""A crop wider than any acquisition, so `SlotReader.cropped` decodes and hands back the FULL frame.

The denominator this measurement needs is the tissue in the whole acquisition, and a stack that has
already been cropped cannot supply it — whatever the crop discarded is exactly what is being counted."""

TIGHT_MM = 80.0
"""The extent knee-1nn's multi-scale reading is gated on, and the one with no margin to spare."""

EXTENTS = (CROP_MM, TIGHT_MM)

SLICES = 12
"""What the cache keeps per series, so the stack measured here is the stack a member would read."""

SAMPLE = 400
"""Studies to decode. The 60 of knee-yrf's original measurement carry a sampling error near a tenth of
the effect they were used to argue for; 400 series-level observations put it near a hundredth, and the
decode is the whole cost so a larger sample buys nothing this decision can use."""

PERCENTILE_LOW = 10


@dataclass(frozen=True)
class CoverageRequest:
    """One study's measurement job, as the pool receives it."""

    identifier: str
    source: str
    planes: dict[str, str]
    slices: int = SLICES
    extents: tuple[float, ...] = EXTENTS


@dataclass(frozen=True)
class CropCoverage:
    """What share of a series' imaged tissue each crop centre keeps, at each physical extent."""

    extents: tuple[float, ...] = field(default=EXTENTS)

    def tissue(self, stack: NDArray[np.float32]) -> NDArray[np.bool_]:
        """The tissue mask, by the same per-slice Otsu rule `TissueCentre` locates with.

        The same rule rather than a second one, so the mask cannot disagree with the centre being scored
        — a centre chosen by one threshold and judged by another measures the two thresholds.
        """
        locator = TissueCentre()
        return np.stack([plane > locator.threshold(plane) for plane in stack])

    def series(self, stack: NDArray[np.float32], spacing: float | None) -> dict[str, float]:
        """Each centre's retained tissue share at each extent, keyed `<centre>@<mm>`.

        Empty when the series holds no tissue at all, or when no extent applies to it — a series whose
        acquisition is narrower than the crop is not evidence about centring, and counting it as perfect
        retention would flatter both centres equally and dilute the comparison.
        """
        mask = self.tissue(stack)
        total = float(mask.sum())
        if total == 0.0:
            return {}
        kept: dict[str, float] = {}
        for name, build in CENTRES.items():
            # Through a reader rather than a copy of the geometry: the box scored is then literally the
            # box a member's read would take, and cannot drift from it.
            reader = SlotReader(centre=build())
            for millimetres in self.extents:
                taken = reader.window(stack, spacing, millimetres)
                if taken is not None:
                    kept[f"{name}@{millimetres:g}"] = float(mask[:, taken[0], taken[1]].sum()) / total
        return kept

    def paired(self, measured: list[dict[str, float]], seed: int = RESAMPLE_SEED) -> list[str]:
        """Tissue minus image at each extent over the SAME series, with a bootstrap interval.

        Paired because the two centres are two readings of one acquisition, not two populations: the
        spread across series is far larger than the difference between centres, so an unpaired
        comparison would drown a real effect in how much tissue each knee happens to show.
        """
        bootstrap = Bootstrap.seeded(seed)
        lines = []
        for millimetres in self.extents:
            tissue, image = f"tissue@{millimetres:g}", f"image@{millimetres:g}"
            both = [kept for kept in measured if tissue in kept and image in kept]
            if not both:
                continue
            deltas = np.array([kept[tissue] - kept[image] for kept in both])
            low, high = bootstrap.mean_interval(deltas)
            star = "*" if low > 0.0 or high < 0.0 else ""
            lines.append(
                f"tissue - image at {millimetres:g} mm: {deltas.mean():+.4f} "
                f"[{low:+.4f}, {high:+.4f}]{star} over {deltas.size} series"
            )
        return lines

    def report(self, measured: list[dict[str, float]], sampled: int, corpus: int) -> list[str]:
        """The retained-tissue distribution per crop, the paired deltas, and WHAT WAS MEASURED ON.

        The denominators are the first line rather than a footnote: a share computed over the series
        that happened to decode is not a share over the corpus, and this project has read one as the
        other three times in a single session.
        """
        lines = [
            f"{len(measured)} series measured, from {sampled} studies sampled of {corpus} in the corpus",
            f"{'crop':16s} {'median':>8s} {'p10':>8s} {'worst':>8s}  {'series':>7s}",
        ]
        for key in sorted({name for kept in measured for name in kept}):
            values = np.array([kept[key] for kept in measured if key in kept])
            lines.append(
                f"{key:16s} {np.median(values):8.4f} {np.percentile(values, PERCENTILE_LOW):8.4f} "
                f"{values.min():8.4f}  {values.size:7d}"
            )
        return lines + self.paired(measured)

    @staticmethod
    def measure_study(request: CoverageRequest) -> list[dict[str, float]]:
        """Every routed slot of one study, measured. A staticmethod so a process pool can pickle it.

        A study that will not decode contributes NOTHING rather than a zero: an unreadable series is
        silent about centring, and counting it as retaining none would charge both centres for the
        DICOM library.
        """
        reader = SlotReader(crop_mm=UNCROPPED)
        pipeline = SlotPipeline(scheme=Routing().built(), reader=reader)
        coverage = CropCoverage(request.extents)
        try:
            study = Study.from_directory(Path(request.source) / request.identifier, request.planes, pipeline)
            routed = list(pipeline.route(study.acquisitions, study.planes).values())
            read = [(reader.cropped(acquisition, request.slices), acquisition.pixel_spacing) for acquisition in routed]
        except (OSError, ValueError, RuntimeError, IndexError) as failure:
            logger.info(f"  {request.identifier} did not decode: {type(failure).__name__}: {failure}")
            return []
        measured = [coverage.series(stack, spacing) for stack, spacing in read if stack is not None]
        return [kept for kept in measured if kept]


def main() -> None:
    """Decode a spread sample of studies uncropped and score both centres at both extents."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=SAMPLE, help="studies to decode")
    parser.add_argument("--slices", type=int, default=SLICES)
    parser.add_argument("--tight-mm", type=float, default=TIGHT_MM, help="the extent with no margin")
    parser.add_argument("--workers", type=int, default=20, help="decode processes")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    split = CorpusSplit(DataRoot.from_config(arguments.paths).raw())
    identifiers = split.identifiers()
    planes = split.planes()
    # Every k-th rather than a random draw: the sample is then a property of the corpus order rather
    # than of a seed, and repeating the measurement cannot quietly answer with different studies.
    chosen = identifiers[:: max(1, len(identifiers) // arguments.sample)][: arguments.sample]

    extents = (CROP_MM, arguments.tight_mm)
    requests = [
        CoverageRequest(
            identifier=name, source=str(split.source), planes=planes, slices=arguments.slices, extents=extents
        )
        for name in chosen
    ]
    measured: list[dict[str, float]] = []
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for kept in pool.map(CropCoverage.measure_study, requests, chunksize=4):
            measured.extend(kept)
    for line in CropCoverage(extents).report(measured, len(chosen), len(identifiers)):
        logger.info(line)


if __name__ == "__main__":
    main()
