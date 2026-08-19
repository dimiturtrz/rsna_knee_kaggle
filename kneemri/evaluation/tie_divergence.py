"""When two series tie for a slot, do they SHOW the same thing? `python -m kneemri.evaluation.tie_divergence`.

`SlotTies` established that one study in five has a slot where two series can both legitimately win,
because `assign` breaks a tie on slice count with `max` (first in LIST order) while the published
members we grafted were routed by a pandas `sort_values` (first in FRAME order). That is an upper
bound on how often the two pipelines CAN diverge, and says nothing about whether the divergence
matters: two runs of the same sequence at the same slice count often differ only in acquisition time,
and swapping them changes nothing at all.

THIS IS THE MEASUREMENT THAT SETTLES IT, and it settles it in the direction that costs nothing to act
on. If the tied candidates are near-identical images, the tie is harmless and the ~0.011 we lose
against the same weights on the same leaderboard lives somewhere else entirely — window policy,
laterality, crop geometry, interpolation — and this hypothesis dies cleanly. If they differ, adopting
frame order makes us agree with every published member by construction.

PIXELS, WHICH IS WHY IT IS NOT IN `slot_ties`. That module reads one DICOM header per series and no
pixel data, so it runs happily beside training; this one decodes two full stacks per contested slot
and does not.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import UInt8
from torch import Tensor

from kneemri.corpus.acquisition import Acquisition
from kneemri.corpus.slot_reader import SlotReader
from kneemri.corpus.slots import SLOTS, SlotScheme
from kneemri.corpus.study import Study

BYTE_RANGE = 255.0
"""What `SlotReader` quantises to, so a distance divided by it reads as a fraction of full scale."""

IDENTICAL = 1.0 / BYTE_RANGE
"""Below one quantisation step apart is the same image, whatever the two pipelines would have picked."""

SAMPLED_SLICES = 12
"""The depth a member reads. The comparison must be at the depth the MODEL sees, not the full stack:
two series can differ in slices the sampler never draws, and that difference cannot reach the head."""


@dataclass(frozen=True)
class SlotDivergence:
    """How far apart the winner and a rival were, for one contested slot of one study.

    Beside its producer rather than in a types module: it exists only to carry this measurement, and
    `distance` means nothing without the normalisation `TieDivergence` applied to it.
    """

    study: str
    slot: str
    winner: str
    rival: str
    distance: float
    """Mean absolute difference over the sampled stack, as a fraction of full scale."""

    @property
    def identical(self) -> bool:
        """Whether the two candidates are the same image to within one quantisation step."""
        return self.distance < IDENTICAL


class TieDivergence:
    """Decodes both candidates for each contested slot and reports how far apart they are."""

    def __init__(self, reader: SlotReader | None = None, scheme: SlotScheme | None = None) -> None:
        self.reader = reader or SlotReader()
        self.scheme = scheme or SlotScheme()

    def of_study(self, directory: Path, planes: dict[str, str]) -> list[SlotDivergence]:
        """Every contested slot of one study, measured. Empty when nothing is contested."""
        study = Study.from_directory(directory, planes)
        divergences = []
        for slot in self.scheme.slots:
            candidates = [
                acquisition
                for acquisition in study.acquisitions
                if slot.accepts(acquisition, planes.get(acquisition.series, ""))
            ]
            divergences.extend(self._of_slot(slot.name, candidates))
        return divergences

    def _of_slot(self, name: str, candidates: list[Acquisition]) -> list[SlotDivergence]:
        """The tied rivals of one slot against its winner, or nothing when the slot is uncontested."""
        if not candidates:
            return []
        # `max` keeps the FIRST maximal element, which is exactly the rule under test — the winner here
        # must be the one `assign` would actually pick, not merely a tied member.
        depth = max(candidate.slice_count for candidate in candidates)
        tied = [candidate for candidate in candidates if candidate.slice_count == depth]
        if len(tied) < 2:  # noqa: PLR2004 - a slot with one maximal candidate is not contested
            return []
        winner, rivals = tied[0], tied[1:]
        read = self._read(winner)
        if read is None:
            return []
        return [
            SlotDivergence(winner.study, name, winner.series, rival.series, distance)
            for rival, distance in ((rival, self._distance(read, rival)) for rival in rivals)
            if distance is not None
        ]

    def _distance(self, read: UInt8[Tensor, "s h w"], rival: Acquisition) -> float | None:
        """Mean absolute difference against a rival, or None when the rival does not decode."""
        other = self._read(rival)
        if other is None or other.shape != read.shape:
            # A shape mismatch is a genuine divergence and not a measurable one: the two series were
            # acquired on different matrices, so no per-pixel difference is defined. Reported as
            # undecidable rather than as zero, which would read as "the tie is harmless".
            return None
        return float(torch.abs(read.float() - other.float()).mean()) / BYTE_RANGE

    def _read(self, acquisition: Acquisition) -> UInt8[Tensor, "s h w"] | None:
        """The candidate as the model would see it, at the depth the model reads."""
        return self.reader.read(acquisition, SAMPLED_SLICES)

    @staticmethod
    def _of(argument: tuple[Path, dict[str, str]]) -> list[SlotDivergence]:
        """Pool worker: one study's contested-slot divergences.

        A staticmethod rather than a module-level function because the house rule wants behaviour on
        the class that owns it — and a staticmethod still pickles by qualified name, which is what the
        child process needs.
        """
        directory, planes = argument
        return TieDivergence().of_study(directory, planes)


@dataclass(frozen=True)
class TieDivergenceReport:
    """The census as prose — the rendering half of what `TieDivergence` used to be.

    Split for the reason the other reports were: measuring decodes two full stacks per contested slot
    across the corpus, and rendering needs only the list they produced. Held together, the census could
    not be exercised without a corpus of DICOMs to decode.
    """

    divergences: list[SlotDivergence]
    studies: int

    def lines(self) -> list[str]:
        """The census, with the verdict each outcome supports stated rather than implied."""
        divergences = self.divergences
        lines = [f"{len(divergences)} contested-slot rivalries over {self.studies} studies"]
        if not divergences:
            return [*lines, "  nothing contested decoded - the hypothesis cannot be tested on this sample"]
        distances = np.array([divergence.distance for divergence in divergences])
        same = sum(divergence.identical for divergence in divergences)
        lines.append(f"  identical to within a quantisation step: {same} of {len(divergences)}")
        lines.append(
            f"  mean |diff| {distances.mean():.4f}  median {np.median(distances):.4f}  max {distances.max():.4f}"
        )
        for slot in SLOTS:
            per_slot = [divergence.distance for divergence in divergences if divergence.slot == slot.name]
            if per_slot:
                lines.append(f"  {slot.name}: {len(per_slot)} rivalries, mean |diff| {np.mean(per_slot):.4f}")
        lines.append(
            "  NEAR ZERO kills the tie-break hypothesis for the 0.011 leaderboard gap: the two routing "
            "rules pick different names for the same picture, and the residual is elsewhere"
        )
        lines.append(
            "  LARGE makes frame order the fix, and it costs nothing - it makes us agree with every "
            "published member by construction rather than by measurement"
        )
        return lines


def main() -> None:
    """Measure how much the tied candidates actually differ, across the training corpus."""
    import argparse  # noqa: PLC0415 - CLI-only, and this module is imported as a library by analyses
    import logging  # noqa: PLC0415 - as above
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415 - as above

    from core.paths import DataRoot  # noqa: PLC0415 - as above
    from kneemri.corpus.study_cache import StudyCache  # noqa: PLC0415 - as above

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=str, default="train_series")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--studies", type=int, default=0, help="0 reads the whole corpus")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    raw = DataRoot.from_config(arguments.paths).raw()
    planes = StudyCache.planes_of(raw / arguments.split.replace("_series", "_series.csv"))
    directories = sorted(path for path in (raw / arguments.split).iterdir() if path.is_dir())
    if arguments.studies:
        directories = directories[: arguments.studies]

    divergences: list[SlotDivergence] = []
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        work = ((directory, planes) for directory in directories)
        for found in pool.map(TieDivergence._of, work, chunksize=4):  # noqa: SLF001
            divergences.extend(found)

    for line in TieDivergenceReport(divergences, len(directories)).lines():
        logger.info("%s", line)


if __name__ == "__main__":
    main()
