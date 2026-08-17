"""What the central slice band actually discards: `python -m kneemri.corpus.band_coverage`.

THE CONSTANT THIS INTERROGATES. `slot_reader.SLICE_BAND = (0.20, 0.80)` carries one line of
justification — "the outermost slices of a knee series are mostly soft tissue outside the joint" — and
nothing has ever measured it. Every member on disk was fitted under it (knee-nas5: `band_low`/`band_high`
appear in none of the six all-ours checkpoints' recorded options), so it is the one reading rule the pool
has never varied, on the decode axis that transfers to the board at 2.4x the usual compression.

WHY IT IS SUSPECTED NOW. On a SAGITTAL series the through-plane axis IS the medial-lateral axis, so this
rule deletes the outer 20% of medial-lateral extent at each end before a model sees a pixel — and
knee-yrg measures Lateral Meniscus (0.7981) as our weakest label of twelve and Lateral OA (0.8215) as
fourth-weakest, with medial beating lateral by +0.0622 and +0.0472. That bead lists "the lateral
compartment sits nearer the edge of a 130 mm crop" as an untested candidate; that is the IN-PLANE crop.
The band is the through-plane one, and it is stronger: a crop keeps the compartment at the edge of the
field, whereas the band removes those slices from the array entirely.

WHY THIS AND NOT AN AUC. Training a band-diverse member costs a fresh 99 GB cache and a GPU slot. What is
decidable without either, and without a label, is how much of a series' imaged tissue lies in the region
the band throws away. That is the constant's own claim, stated as a number: if the discarded ends hold 2%
of the tissue they are air and the rule is right; if they hold 15% the rule is deleting knee.

WHAT IT CANNOT SAY, stated because `crop_coverage` had to state the same. Discarded TISSUE is not
discarded JOINT — the outermost sagittal slices hold thigh and calf as well as compartment. This bounds
the claim from one side only: a band whose discarded ends are empty cannot be hiding the lateral
meniscus, and that is the direction that kills the hypothesis cheaply. A band whose ends are full is a
reason to build the cache, not a result.
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from core.paths import DataRoot
from kneemri.corpus.crop_centre import TissueCentre
from kneemri.corpus.slot_reader import SLICE_BAND, Band, SlotReader
from kneemri.corpus.slots import Routing
from kneemri.corpus.study import SlotPipeline, Study
from kneemri.corpus.study_cache import CorpusSplit

logger = logging.getLogger(__name__)

FULL_BAND = Band(low=0.0, high=1.0)
"""The whole series, which is the only denominator that can price what a narrower band removes.

A stack already sampled over the central band cannot answer this: what it dropped is exactly what is
being counted.
"""

DEPTH = 24
"""Slices read per series — twice what a member reads.

The measurement is a profile ALONG the through-plane axis, so its resolution is what bounds the answer,
not the twelve a cache happens to keep. Twenty-four puts each end of the 20% tail at roughly five slices,
which is enough for the tail's mass to be a mean rather than one slice's luck.
"""

SAMPLE = 400
"""Studies to decode, matching `crop_coverage` — a sampling error near a hundredth of the effect."""

RESAMPLES = 1500
RESAMPLE_SEED = 11
INTERVAL = (2.5, 97.5)

PERCENTILE_HIGH = 90


@dataclass(frozen=True)
class BandRequest:
    """One study's measurement job, as the pool receives it."""

    identifier: str
    source: str
    planes: dict[str, str]
    depth: int = DEPTH


@dataclass(frozen=True)
class SlotShares:
    """One slot's measured series, as the columns a report row is made of."""

    slot: str
    discarded: NDArray[np.float64]
    low_end: NDArray[np.float64]
    high_end: NDArray[np.float64]
    span_mm: float

    @classmethod
    def of(cls, slot: str, measured: list[tuple[str, dict[str, float]]]) -> "SlotShares":
        """The rows belonging to one slot, gathered once rather than re-filtered per column."""
        mine = [one for name, one in measured if name == slot]
        spans = [one["span_mm"] for one in mine if one["span_mm"] > 0.0]
        return cls(
            slot=slot,
            discarded=np.array([one["discarded"] for one in mine]),
            low_end=np.array([one["low_end"] for one in mine]),
            high_end=np.array([one["high_end"] for one in mine]),
            # Zero where no series in this slot carried a spacing tag: an unknown extent reports as
            # unknown rather than as a millimetre nobody measured.
            span_mm=float(np.mean(spans)) if spans else 0.0,
        )

    def row(self, coverage: "BandCoverage") -> str:
        """The formatted line, with the cut restated as the distance it removes from each end."""
        low, high = coverage.interval(self.discarded)
        return (
            f"{self.slot:16s} {self.discarded.mean():10.4f} [{low:+.4f}, {high:+.4f}] "
            f"{np.percentile(self.discarded, PERCENTILE_HIGH):7.4f} "
            f"{self.low_end.mean():8.4f} {self.high_end.mean():9.4f} "
            f"{self.span_mm:8.1f} {self.span_mm * coverage.band.low:8.1f} {self.discarded.size:6d}"
        )


@dataclass(frozen=True)
class BandCoverage:
    """What share of a series' imaged tissue a central band throws away, per anatomical plane."""

    band: Band = Band()

    def profile(self, stack: NDArray[np.float32]) -> NDArray[np.float64]:
        """Tissue mass per slice, against ONE Otsu threshold taken over the whole series.

        Otsu is `TissueCentre`'s rule, but applied to the stack rather than to each slice, and the
        difference is load-bearing here. `TissueCentre` judges slices independently because it asks a
        WITHIN-slice question (where is this slice's centroid), and a near-empty slice abstains via
        `MIN_TISSUE_PIXELS`. This asks a BETWEEN-slice one — is there more tissue here than there — and a
        per-slice threshold makes that question unanswerable: an end slice holding only background gets
        its own histogram split, so its noise is thresholded into "tissue" and the tail inflates. One
        threshold for the series is the only reading under which an empty end weighs nothing.
        """
        threshold = TissueCentre().threshold(stack)
        return np.array([float((plane > threshold).sum()) for plane in stack])

    def discarded(self, stack: NDArray[np.float32]) -> dict[str, float]:
        """The share of tissue outside the band, and the share at each end separately.

        Both ends separately because they are not interchangeable on a sagittal series: one is medial and
        one is lateral, and an asymmetry there is itself evidence about what the rule removes.

        Empty for a series holding no tissue at all — such a series is silent about the band rather than
        evidence that it discards nothing.
        """
        mass = self.profile(stack)
        total = float(mass.sum())
        if total == 0.0:
            return {}
        position = np.linspace(0.0, 1.0, mass.size)
        low = float(mass[position < self.band.low].sum()) / total
        high = float(mass[position > self.band.high].sum()) / total
        return {"low_end": low, "high_end": high, "discarded": low + high}

    def interval(self, values: NDArray[np.float64], seed: int = RESAMPLE_SEED) -> tuple[float, float]:
        """A bootstrap interval on the mean, so a share is never reported as a bare point estimate."""
        generator = np.random.default_rng(seed)
        resampled = [float(generator.choice(values, values.size).mean()) for _ in range(RESAMPLES)]
        low, high = np.percentile(resampled, INTERVAL)
        return float(low), float(high)

    def report(self, measured: list[tuple[str, dict[str, float]]], sampled: int, corpus: int) -> list[str]:
        """Discarded tissue per slot, with the DENOMINATORS on the first line rather than in a footnote.

        A share computed over the series that happened to decode is not a share over the corpus, and this
        project has read one as the other three times in a single session.
        """
        header = [
            f"band {self.band.low:g}-{self.band.high:g} over {len(measured)} series "
            f"from {sampled} studies sampled of {corpus} in the corpus",
            f"uniform tissue along the axis would discard {1.0 - (self.band.high - self.band.low):.4f} — "
            f"a share BELOW that is the constant's claim, at or above it is knee",
            f"{'slot':16s} {'discarded':>10s} {'95% CI':>18s} {'p90':>7s} {'low end':>8s} {'high end':>9s} "
            f"{'span mm':>8s} {'cut/end':>8s} {'n':>6s}",
        ]
        slots = sorted({name for name, _ in measured})
        return header + [SlotShares.of(slot, measured).row(self) for slot in slots]

    @staticmethod
    def measure_study(request: BandRequest) -> list[tuple[str, dict[str, float]]]:
        """Every routed slot of one study, read over the FULL series and scored against the band.

        A study that will not decode contributes NOTHING rather than a zero, as in `crop_coverage`: an
        unreadable series is silent about the band, and counting it as discarding none would charge the
        rule for the DICOM library.
        """
        reader = SlotReader(band=FULL_BAND)
        pipeline = SlotPipeline(scheme=Routing().built(), reader=reader)
        coverage = BandCoverage()
        try:
            study = Study.from_directory(Path(request.source) / request.identifier, request.planes, pipeline)
            routed = pipeline.route(study.acquisitions, study.planes)
            read = [
                (slot, (acquisition, reader.cropped(acquisition, request.depth)))
                for slot, acquisition in routed.items()
            ]
        except (OSError, ValueError, RuntimeError, IndexError) as failure:
            logger.info(f"  {request.identifier} did not decode: {type(failure).__name__}: {failure}")
            return []
        scored = [
            (slot, coverage.discarded(stack) | {"span_mm": acquisition.through_plane_mm or 0.0})
            for slot, (acquisition, stack) in read
            if stack is not None
        ]
        return [(slot, one) for slot, one in scored if len(one) > 1]


def main() -> None:
    """Decode a spread sample of studies over the whole series and price what the band removes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=SAMPLE, help="studies to decode")
    parser.add_argument("--depth", type=int, default=DEPTH, help="slices read along the through-plane axis")
    parser.add_argument("--band-low", type=float, default=SLICE_BAND[0])
    parser.add_argument("--band-high", type=float, default=SLICE_BAND[1])
    parser.add_argument("--workers", type=int, default=20, help="decode processes")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    split = CorpusSplit(DataRoot.from_config(arguments.paths).raw())
    identifiers = split.identifiers()
    planes = split.planes()
    # Every k-th rather than a random draw, as `crop_coverage` does: the sample is a property of the
    # corpus order rather than of a seed, so repeating this cannot quietly answer with other studies.
    chosen = identifiers[:: max(1, len(identifiers) // arguments.sample)][: arguments.sample]

    requests = [
        BandRequest(identifier=name, source=str(split.source), planes=planes, depth=arguments.depth) for name in chosen
    ]
    measured: list[tuple[str, dict[str, float]]] = []
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for scored in pool.map(BandCoverage.measure_study, requests, chunksize=4):
            measured.extend(scored)
    band = Band(low=arguments.band_low, high=arguments.band_high)
    for line in BandCoverage(band).report(measured, len(chosen), len(identifiers)):
        logger.info(line)


if __name__ == "__main__":
    main()
