"""Does keeping 12 slices of a deep stack cost accuracy? `python -m kneemri.evaluation.stack_coverage`.

THE HYPOTHESIS THIS TESTS. The cache keeps 12 slices per series against a median of about 30 acquired
(knee-ezz), so a study scanned at 40 slices has three quarters of its imaging discarded before the model
sees anything, while a study scanned at 14 loses almost none. If that discarding costs accuracy, a
finding's AUC should fall as the acquired stack gets deeper — and it should fall for the findings whose
bottleneck is the PIXELS, not for the ones whose bottleneck is the label reader (`supervision_gap`).

WHY IT IS WORTH ASKING HERE RATHER THAN BY TRAINING. Depth spread across windows measured 0.036
(knee-ghd), larger than every method change this project has made put together, and whole-stack reading
collected some of it. What remains unmeasured is the loss that happens BEFORE any of that: the cache
build itself, which no reading policy can undo. This asks the question with arithmetic over predictions
that already exist and file counts that cost no DICOM parsing.

WHAT WOULD REFUTE IT: a flat profile across depth, or one that moves for the label-bottlenecked findings
too — the second would mean depth is standing in for something else about how a study was scanned, since
no amount of imaging can fix a mislabelled report.

READ THE CONFOUND FIRST. Deeper stacks are not a random subset of studies: scanner, protocol and body
habitus all travel with slice count, and a deeper scan may simply be a better scan. This measures an
ASSOCIATION and says so; it cannot separate "we threw away more" from "these studies differ".
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core.bootstrap import INTERVAL, RESAMPLE_SEED, RESAMPLES
from core.findings import TARGETS
from kneemri.evaluation.scores import LabelScores

logger = logging.getLogger(__name__)

CACHED_SLICES = 12
"""What the cache keeps per series, whatever was acquired."""

PRESENT = 0.5
"""Where a graded target becomes a positive, matching LabelScores."""

GROUPS = 3
"""Quantile bands of acquired depth. Three keeps roughly 1,400 studies per band, which is enough for a
per-label AUC to mean something; twelve labels over five bands would be reading noise."""


class StackCoverage:
    """How much of each study's acquired stack survives the cache, and what it costs."""

    def __init__(self, scores: LabelScores | None = None) -> None:
        self.scores = scores or LabelScores()

    def acquired(self, series_root: Path, identifiers: list[str]) -> dict[str, int]:
        """The deepest series each study acquired, by COUNTING FILES rather than reading headers.

        One slice is one file here, so a directory listing answers what a DICOM parse would — over 4,407
        studies that is the difference between seconds and many minutes, and nothing downstream needs
        the header. The deepest series is the right summary because it is the one a slot most often
        wins: `SlotScheme.assign` breaks ties toward the densest stack.
        """
        depths: dict[str, int] = {}
        for name in identifiers:
            study = series_root / name
            if not study.is_dir():
                continue
            counts = [sum(1 for _ in series.iterdir()) for series in study.iterdir() if series.is_dir()]
            if counts:
                depths[name] = max(counts)
        return depths

    def bands(self, depths: dict[str, int], identifiers: list[str], groups: int = GROUPS) -> list[list[int]]:
        """Row indices split into equal-sized bands by acquired depth, shallowest first.

        Quantiles rather than fixed thresholds, because the point is to compare like-sized populations —
        a fixed cut at 30 slices would put most of the corpus in one band and read the rest as noise.
        """
        known = [(index, depths[name]) for index, name in enumerate(identifiers) if name in depths]
        known.sort(key=lambda pair: pair[1])
        return [[index for index, _ in band] for band in np.array_split(np.array(known, dtype=object), groups)]

    def trend(
        self,
        predictions: Float[np.ndarray, "studies targets"],
        truth: Float[np.ndarray, "studies targets"],
        bands: list[list[int]],
        seed: int = RESAMPLE_SEED,
    ) -> dict[str, tuple[list[float], float, float]]:
        """Each finding's AUC per band, and the deepest-minus-shallowest delta WITH its interval.

        The interval is not decoration and this method exists so the delta cannot be reported without
        it. A per-band AUC over a third of the corpus carries a sampling error near 0.02, so a 0.04
        trend read off the bare numbers looks like a finding and is not one — which is exactly the
        mistake this project has made often enough to legislate against.

        Bootstrapped over STUDIES within each band, independently, because the two bands are different
        studies rather than paired measurements of the same ones.
        """
        generator = np.random.default_rng(seed)
        shallow, deep = np.array(bands[0], dtype=int), np.array(bands[-1], dtype=int)
        measured: dict[str, tuple[list[float], float, float]] = {}
        for column, target in enumerate(TARGETS):
            per_band = [
                self.scores.area_under_curve(
                    predictions[np.array(band, dtype=int), column],
                    (truth[np.array(band, dtype=int), column] > PRESENT).astype(np.int64),
                )
                for band in bands
            ]
            resampled = [
                self._delta(
                    predictions,
                    truth,
                    column,
                    generator.choice(deep, len(deep)),
                    generator.choice(shallow, len(shallow)),
                )
                for _ in range(RESAMPLES)
            ]
            low, high = (float(bound) for bound in np.nanpercentile(resampled, INTERVAL))
            measured[target] = (per_band, low, high)
        return measured

    def _delta(
        self,
        predictions: Float[np.ndarray, "studies targets"],
        truth: Float[np.ndarray, "studies targets"],
        column: int,
        deep: NDArray[np.int64],
        shallow: NDArray[np.int64],
    ) -> float:
        """One finding's deep-minus-shallow AUC over two given index sets."""
        return self.scores.area_under_curve(
            predictions[deep, column], (truth[deep, column] > PRESENT).astype(np.int64)
        ) - self.scores.area_under_curve(
            predictions[shallow, column], (truth[shallow, column] > PRESENT).astype(np.int64)
        )


@dataclass(frozen=True)
class StackCoverageReport:
    """The measured trend as a table — the rendering half of what `StackCoverage` used to be.

    Split for the reason the diagnostics were: measuring needs the series on disk and the published
    predictions, rendering needs the two dicts they produced, and holding them together made the table
    untestable without the corpus.
    """

    measured: dict[str, tuple[list[float], float, float]]
    bands: list[list[int]]

    def lines(self) -> list[str]:
        """Every finding's bands and its trend, starred only where the interval excludes zero.

        The count of stars is printed against what CHANCE predicts, because twelve intervals at 95%
        produce about one false positive by construction and a reader who forgets that will find a
        result in any table this shape.
        """
        measured, bands = self.measured, self.bands
        ordered = sorted(TARGETS, key=lambda target: measured[target][0][-1] - measured[target][0][0])
        header = " ".join(f"{f'band{index + 1}':>8s}" for index in range(len(bands)))
        lines = [
            f"{len(bands)} bands of {len(bands[0])} studies",
            f"{'finding':20s} {header}   {'deep-shallow':>12s}  95% interval",
        ]
        cleared = 0
        for target in ordered:
            per_band, low, high = measured[target]
            delta = per_band[-1] - per_band[0]
            star = "*" if low > 0.0 or high < 0.0 else ""
            cleared += bool(star)
            lines.append(
                f"{target:20s} "
                + " ".join(f"{value:8.4f}" for value in per_band)
                + f"   {delta:+12.4f}  [{low:+.4f}, {high:+.4f}]{star}"
            )
        lines.append(f"{cleared} of {len(TARGETS)} clear zero; chance alone predicts about {0.05 * len(TARGETS):.1f}")
        return lines


def main() -> None:
    """Score the published members' held-out predictions inside bands of acquired stack depth."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=GROUPS)
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from core.data.corpus.study_cache import CorpusSplit  # noqa: PLC0415 - CLI-only
    from core.data.labels.targets import Targets  # noqa: PLC0415 - CLI-only
    from core.paths import DataRoot  # noqa: PLC0415 - CLI-only
    from kneemri.evaluation.published_ruler import PublishedRuler  # noqa: PLC0415 - CLI-only
    from kneemri.inference.vote import RankAverage  # noqa: PLC0415 - CLI-only

    root = DataRoot.from_config(arguments.paths)
    split = CorpusSplit(root.raw())
    names, per_seed = PublishedRuler(root.processed() / "published").per_seed()
    combined = RankAverage().combine(per_seed)
    truth = Targets.load(arguments.labels, split.studies_table, names).scores

    coverage = StackCoverage()
    depths = coverage.acquired(split.source, names)
    counted = np.array(sorted(depths.values()))
    logger.info(
        "acquired depth: median %d, quartiles %d/%d, and the cache keeps %d",
        int(np.median(counted)),
        int(np.percentile(counted, 25)),
        int(np.percentile(counted, 75)),
        CACHED_SLICES,
    )
    bands = coverage.bands(depths, names, arguments.groups)
    for line in StackCoverageReport(coverage.trend(combined, truth, bands), bands).lines():
        logger.info(line)


if __name__ == "__main__":
    main()
