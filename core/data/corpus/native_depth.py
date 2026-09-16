"""How many slices a routed slot ACTUALLY holds, and at what spacing:
`python -m core.data.corpus.native_depth`.

WHAT IT DECIDES. Every slice-sampling argument this project has made — the band, the density square,
the window count — names a number of slices to READ, and none of them could say how many were
ACQUIRED. Without that the two are indistinguishable: reading 20 slices of a 30-slice series samples
what the scanner recorded, and reading 80 of the same series interpolates it 2.7x. Those are different
mechanisms with the same units, and a gain measured on the second one is not a resolution result.

WHY IT WAS BUILT, 2026-09-16. The competition forum carries a monotone series from the Raptor weights
author (topic 737696, his own board, his own pipeline): 44 slices 0.917, 64 slices 0.928, 80 slices
0.932 over a 0.02-0.98 span, with the conclusion "density is the lever rather than span, and 1.20
percent of the stack per slice is the finest I've measured and also the best". Read against this
measurement 1.20 percent of the stack is 1.25 mm, and the corpus is acquired at 3.5 mm — so the
finest point in that series holds no information the coarsest one lacks, and whatever it buys is not
through-plane resolution. See knee-3kpv9.

WHAT IT CANNOT SAY. Nothing about whether interpolating is WORTH it. Resampling a stack changes what
a pooled reading averages over, and more reads of the same voxels is the TTA axis, which this repo
priced separately (knee-eeo). This bounds the claim from one side only, which is the same shape as
`band_coverage` and `crop_coverage`: it says what information the DICOMs contain, never what a model
does with it.

HEADERS AND ROUTING ONLY — no pixels are decoded, so it costs a minute rather than a cache rebuild.
"""

import logging
import statistics
from dataclasses import dataclass

from core import cli
from core.data.corpus.audit import AuditRequest, CorpusAudit
from core.data.corpus.study_cache import CorpusSample

logger = logging.getLogger(__name__)

QUANTILE = 10
"""The tail read at each end, as a denominator: p10 and p90 rather than a min and a max.

A min over four hundred studies is one series, and a slice budget chosen to cover the single deepest
acquisition in the sample is a budget chosen by an outlier.
"""


@dataclass(frozen=True)
class SlotDepth:
    """One routed slot's acquired depth, in slices and in millimetres."""

    slot: str
    slices: int
    spacing_mm: float

    @property
    def span_mm(self) -> float:
        """The through-plane extent this series actually imaged."""
        return self.slices * self.spacing_mm


@dataclass(frozen=True)
class NativeDepth:
    """The acquired depth of every routed slot, sampled across the corpus."""

    quantile: int = QUANTILE

    @staticmethod
    def measure_study(request: AuditRequest) -> list[SlotDepth]:
        """Every routed slot of one study, read from headers alone.

        A series with no spacing tag is dropped rather than read as zero millimetres: an untagged
        acquisition is silent about depth in mm, though its slice COUNT would still be true.
        """
        return [
            SlotDepth(slot=slot, slices=acquisition.slice_count, spacing_mm=spacing)
            for slot, acquisition in request.routed().slots.items()
            if (spacing := acquisition.slice_spacing) is not None
        ]

    def row(self, slot: str, measured: list[SlotDepth]) -> str:
        """One slot's line: the depth distribution, and the millimetres a given read would land on."""
        depths = sorted(one.slices for one in measured if one.slot == slot)
        spacings = sorted(one.spacing_mm for one in measured if one.slot == slot)
        median_slices = statistics.median(depths)
        median_spacing = statistics.median(spacings)
        low = depths[len(depths) * self.quantile // 100]
        high = depths[len(depths) * (100 - self.quantile) // 100]
        return (
            f"{slot:16s} {len(depths):6d} {low:8d} {median_slices:8.0f} {high:6d} "
            f"{median_spacing:11.2f} {median_slices * median_spacing:9.1f}"
        )

    def report(self, measured: list[SlotDepth], sample: CorpusSample) -> list[str]:
        """Acquired depth per slot, with the DENOMINATORS on the first line rather than in a footnote."""
        header = [
            f"native depth: {sample.line(len(measured), 'series')}",
            "a read DEEPER than the median interpolates; the mm column is what one slice of that read covers",
            f"{'slot':16s} {'n':>6s} {'p' + str(self.quantile):>8s} {'median':>8s} "
            f"{'p' + str(100 - self.quantile):>6s} {'spacing mm':>11s} {'span mm':>9s}",
        ]
        slots = sorted({one.slot for one in measured})
        return header + [self.row(slot, measured) for slot in slots]


AUDIT = CorpusAudit(unit="series", workers=16)
"""Headers and routing only, at the reader every member reads at — so no geometry flag can apply."""


def main() -> None:
    """Route a spread sample of studies and report how deep each slot was actually acquired."""
    command = cli.Command(__doc__, logger)
    AUDIT.flags(command)
    arguments = command.parsed()

    sample, requests = AUDIT.drawn(command, arguments)
    measured = AUDIT.mapped(NativeDepth.measure_study, requests, arguments.workers)
    AUDIT.reported(command, NativeDepth().report(measured, sample), sample, len(measured))


if __name__ == "__main__":
    main()
