"""Which crop centre actually keeps the knee: `python -m core.data.corpus.crop_coverage`.

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

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from core import cli
from core.bootstrap import RESAMPLE_SEED, Bootstrap
from core.data.corpus.audit import UNCROPPED, AuditRequest, CorpusAudit, Reading
from core.data.corpus.crop_centre import CENTRES, CROP_MM, TissueCentre
from core.data.corpus.slot_reader import SlotReader
from core.data.corpus.study_cache import CorpusSample

logger = logging.getLogger(__name__)

TIGHT_MM = 80.0
"""The extent knee-1nn's multi-scale reading is gated on, and the one with no margin to spare."""

EXTENTS = (CROP_MM, TIGHT_MM)

SLICES = 12
"""What the cache keeps per series, so the stack measured here is the stack a member would read."""

PERCENTILE_LOW = 10

AUDIT = CorpusAudit(reading=Reading(crop_mm=UNCROPPED), unit="series", slices=SLICES)
"""Read UNCROPPED, because the denominator is the tissue in the whole acquisition.

The default sample is the harness's 400, which is what `crop_alignment` and `slot_geometry` also draw —
the three readings compose only because all three take the same every-k-th studies. knee-yrf's original
60 carried a sampling error near a tenth of the effect it was used to argue for.
"""


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

    def report(self, measured: list[dict[str, float]], sample: CorpusSample) -> list[str]:
        """The retained-tissue distribution per crop, the paired deltas, and WHAT WAS MEASURED ON.

        The denominators are the first line rather than a footnote: a share computed over the series
        that happened to decode is not a share over the corpus, and this project has read one as the
        other three times in a single session.
        """
        lines = [
            sample.line(len(measured), "series"),
            f"{'crop':16s} {'median':>8s} {'p10':>8s} {'worst':>8s}  {'series':>7s}",
        ]
        for key in sorted({name for kept in measured for name in kept}):
            values = np.array([kept[key] for kept in measured if key in kept])
            lines.append(
                f"{key:16s} {np.median(values):8.4f} {np.percentile(values, PERCENTILE_LOW):8.4f} "
                f"{values.min():8.4f}  {values.size:7d}"
            )
        return lines + self.paired(measured)

    def measure_study(self, request: AuditRequest) -> list[dict[str, float]]:
        """Every routed slot of one study, scored at THIS instance's extents.

        An instance method because the extents are what the pool has to carry, and a frozen dataclass of
        one tuple pickles. A study that will not decode contributes NOTHING rather than a zero: an
        unreadable series is silent about centring, and counting it as retaining none would charge both
        centres for the DICOM library.
        """
        measured = [
            self.series(stack, acquisition.pixel_spacing)
            for _, acquisition, stack in request.routed().decoded(request.slices)
            if stack is not None
        ]
        return [kept for kept in measured if kept]


def main() -> None:
    """Decode a spread sample of studies uncropped and score both centres at both extents."""
    command = cli.Command(__doc__, logger)
    AUDIT.flags(command)
    command.parser.add_argument("--slices", type=int, default=SLICES)
    command.parser.add_argument("--tight-mm", type=float, default=TIGHT_MM, help="the extent with no margin")
    arguments = command.parsed()

    audit = AUDIT.with_slices(arguments.slices)
    coverage = CropCoverage((CROP_MM, arguments.tight_mm))
    sample, requests = audit.drawn(command, arguments)
    measured = audit.mapped(coverage.measure_study, requests, arguments.workers)
    audit.reported(command, coverage.report(measured, sample), sample, len(measured))


if __name__ == "__main__":
    main()
