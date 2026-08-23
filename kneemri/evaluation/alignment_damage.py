"""Does cross-slot misregistration cost AUC, and only on the findings small enough to care:
`python -m kneemri.evaluation.alignment_damage`.

THE GATE THIS IS. `crop_alignment` measured that a study's routed slots disagree about where the knee
sits by 5.7 mm at the median and 10.7 mm at the tail, after roughly two thirds of the raw 9-14 mm
displacement turns out to be common mode and therefore harmless to slot correspondence. At 0.387 mm/px
the residual is about fifteen pixels, against a meniscal tear of 1-2 mm. That is a MEASUREMENT OF THE
INPUT and it says nothing yet about the output. Both fixes it licenses — a tissue-centred crop and a
slot-to-slot registration — cost a 99 GB cache rebuild and hours of refit, so something cheaper has to
decide whether either is worth starting. This is that thing: it spends no GPU and it can only kill.

THE PREDICTION, AND IT IS NOT INVENTED HERE. `core.data.corpus.anatomy.Extent` already partitions the
twelve findings into FOCAL and REGIONAL, assigned from the STRUCTURE and — by its own docstring — never
from a measured per-label result, precisely so that a later per-label pattern stays falsifiable. It
already states the two-sided consequence: focal findings are the ones "a tighter, finer crop can only
help", regional ones are centimetre-scale and "resolution buys these nothing". If misregistration is
doing damage, studies whose slots disagree MORE should score worse on the seven focal findings and not
on the five regional ones.

WHY THE DIFFERENTIAL IS THE READ AND THE RAW GAP IS NOT. A study with a scattered tissue centroid is
plausibly just a WORSE STUDY — motion, poor coil placement, a partial acquisition — and a worse study
scores lower on everything. That confound moves focal and regional together. Misregistration moves only
the focal half. So the quantity that survives the confound is the DIFFERENCE OF DIFFERENCES: the focal
low-minus-high gap minus the regional one. A large raw gap with a zero differential is a difficulty
effect wearing a registration effect's clothes, and it licenses no rebuild.

WHAT A NULL WOULD NOT RULE OUT. That the crop is mis-centred in the common mode, which this cannot see
at all — every study shares it, so it cannot separate studies and cannot appear as a spread effect. It
would kill the REGISTRATION fix on this evidence, not the CENTRING one, which knee-yrf priced on
retention instead. Nor would it clear the proxy: an Otsu mass centroid is not a joint centre, and a
spread statistic built on it is noisy in a way that shrinks any true gap toward zero.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from core.data.corpus.anatomy import ANATOMY, Extent
from core.findings import TARGETS
from kneemri.evaluation.scores import BOOTSTRAP_SEED, LabelScores

logger = logging.getLogger(__name__)

RESAMPLES = 2000
"""Bootstrap draws for the differential's interval. The differential is a difference of four AUCs, so
it carries more sampling error than any of them and is the one number here that must not be read bare."""

INTERVAL = (2.5, 97.5)
"""A 95% interval, two-sided, because the prediction is signed but the null is not."""

MIDPOINT = 0.5
"""Graded targets binarise here, matching `LabelScores.per_label` — an AUC needs two classes, and this
is the same threshold every other ruler in the project uses, not a choice made for this analysis."""


@dataclass(frozen=True)
class SpreadSplit:
    """The holdout, cut in half at its median cross-slot spread, with both sides named.

    A MEDIAN split rather than a threshold in millimetres. The millimetre scale is the proxy's, not the
    anatomy's — an Otsu centroid's absolute value has no calibrated meaning — so any fixed cut would be a
    magic number dressed as a physical one. The median guarantees both halves are the same size, which is
    what keeps the two AUCs comparable at equal sampling error.
    """

    low: NDArray[np.bool_]
    high: NDArray[np.bool_]
    boundary: float

    def sizes(self) -> tuple[int, int]:
        """How many studies fell each side, which is the denominator of every number below it."""
        return int(self.low.sum()), int(self.high.sum())


@dataclass(frozen=True)
class AlignmentDamage:
    """Whether a study's slot disagreement predicts its error, separately for focal and regional findings."""

    scores: LabelScores = LabelScores()

    def split(self, spread: NDArray[np.float32]) -> SpreadSplit:
        """Cut the studies at their median spread, abstentions excluded from both halves.

        A study with one routed slot carries nan and belongs to neither side. Assigning it to the low
        half would read "no disagreement measurable" as "no disagreement", which is the same error as
        scoring an undefined AUC at 0.5.
        """
        measurable = ~np.isnan(spread)
        boundary = float(np.median(spread[measurable]))
        return SpreadSplit(measurable & (spread <= boundary), measurable & (spread > boundary), boundary)

    def label_gaps(
        self,
        predictions: NDArray[np.float64],
        targets: NDArray[np.float32],
        split: SpreadSplit,
    ) -> dict[str, float]:
        """Per finding, the AUC on well-agreeing studies minus the AUC on poorly-agreeing ones.

        Positive means the finding is scored better where the slots agree, which is the direction
        misregistration damage would take.
        """
        low = self.scores.per_label(predictions[split.low], targets[split.low])
        high = self.scores.per_label(predictions[split.high], targets[split.high])
        return {target: low[target] - high[target] for target in TARGETS}

    def by_extent(self, gaps: dict[str, float]) -> dict[Extent, float]:
        """The mean gap within each of the two structural classes, nan-safe.

        `nanmean`, because a finding with one class present on a half is undefined there rather than
        zero, and letting that poison its whole class would report a missing label as a null effect.
        """
        return {
            extent: float(np.nanmean([gaps[target] for target in TARGETS if ANATOMY[target].extent == extent]))
            for extent in Extent
        }

    def differential(self, grouped: dict[Extent, float]) -> float:
        """Focal gap minus regional gap: the part a general-difficulty confound cannot produce."""
        return grouped[Extent.FOCAL] - grouped[Extent.REGIONAL]

    def interval(
        self,
        predictions: NDArray[np.float64],
        targets: NDArray[np.float32],
        spread: NDArray[np.float32],
        resamples: int = RESAMPLES,
    ) -> tuple[float, float]:
        """The differential's 95% interval, by bootstrap over studies.

        The split is RECOMPUTED inside each draw rather than fixed once. Its median boundary is itself
        estimated from the sample, and holding it fixed would quote an interval for a differential
        conditional on a boundary that the data chose — narrower than the truth, by exactly the amount
        the boundary's own error contributes.
        """
        generator = np.random.default_rng(BOOTSTRAP_SEED)
        studies = targets.shape[0]
        drawn: list[float] = []
        for _ in range(resamples):
            index = generator.integers(0, studies, size=studies)
            value = self.differential(
                self.by_extent(self.label_gaps(predictions[index], targets[index], self.split(spread[index])))
            )
            if not np.isnan(value):
                drawn.append(value)
        if not drawn:
            return (float("nan"), float("nan"))
        return (float(np.percentile(drawn, INTERVAL[0])), float(np.percentile(drawn, INTERVAL[1])))

    def report(
        self,
        predictions: NDArray[np.float64],
        targets: NDArray[np.float32],
        spread: NDArray[np.float32],
        resamples: int = RESAMPLES,
    ) -> list[str]:
        """Every number this analysis produces, with the denominators it was computed over."""
        split = self.split(spread)
        low, high = split.sizes()
        gaps = self.label_gaps(predictions, targets, split)
        grouped = self.by_extent(gaps)
        value = self.differential(grouped)
        floor, ceiling = self.interval(predictions, targets, spread, resamples)
        lines = [
            f"{low} well-agreeing studies and {high} poorly-agreeing, split at {split.boundary:.2f} mm "
            f"of mean pairwise cross-slot spread; {int(np.isnan(spread).sum())} abstained with one slot",
            f"{'finding':18s} {'extent':10s} {'AUC gap low-high':>17s}",
        ]
        lines.extend(f"{target:18s} {ANATOMY[target].extent.value:10s} {gaps[target]:17.4f}" for target in TARGETS)
        lines.extend(
            [
                f"focal mean gap {grouped[Extent.FOCAL]:+.4f}, regional mean gap {grouped[Extent.REGIONAL]:+.4f}",
                f"DIFFERENTIAL (focal minus regional) {value:+.4f}, 95% [{floor:+.4f}, {ceiling:+.4f}]",
            ]
        )
        return lines


@dataclass(frozen=True)
class HoldoutJoin:
    """Reads the two keyed records this analysis needs and puts them in one row order.

    A class rather than two loose functions because the whole reason both exist is that a prediction, a
    truth and a spread must end up indexed the same way; keeping them apart is what would let one of the
    three drift onto a different row order without anything noticing.
    """

    def combined(self, members: list[Path]) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.str_]]:
        """Rank-average the named members' holdout records, on the studies ALL of them scored.

        The intersection rather than the union, and the members are aligned by IDENTIFIER rather than by row
        order. Two records written by different runs are not guaranteed to hold the same studies in the same
        order, and averaging them positionally would silently mix one study's prediction with another's
        truth — which reads as a weak model rather than as the bug it is.
        """
        loaded = [np.load(path, allow_pickle=True) for path in members]
        shared = sorted(set.intersection(*[set(record["studies"].tolist()) for record in loaded]))
        stacked = []
        for record in loaded:
            order = {name: row for row, name in enumerate(record["studies"].tolist())}
            rows = np.array([order[name] for name in shared])
            values = record["predictions"][rows].astype(np.float64)
            stacked.append(np.argsort(np.argsort(values, axis=0), axis=0) / max(1, len(shared) - 1))
        first = loaded[0]
        truth_order = {name: row for row, name in enumerate(first["studies"].tolist())}
        truth = first["truth"][np.array([truth_order[name] for name in shared])]
        return np.mean(stacked, axis=0), truth, np.array(shared)

    def aligned(self, spread_file: Path, studies: NDArray[np.str_]) -> NDArray[np.float32]:
        """This run's per-study spread, put in the same row order as the holdout records.

        A study the alignment pass never decoded comes back nan, which `split` then abstains on. It is not
        dropped here, because dropping it would change the row count the predictions were built with and
        every index below would be against a different denominator.
        """
        record = np.load(spread_file, allow_pickle=True)
        measured = dict(zip(record["studies"].tolist(), record["spread"].tolist(), strict=True))
        return np.array([measured.get(name, np.nan) for name in studies.tolist()], dtype=np.float32)


def main() -> None:
    """Join a spread dump against a member mix and print whether disagreement costs the focal findings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spread", type=Path, required=True, help="the .npz written by crop_alignment --dump")
    parser.add_argument("--members", type=Path, nargs="+", required=True, help="holdout .npz records to mix")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    join = HoldoutJoin()
    predictions, truth, studies = join.combined(arguments.members)
    spread = join.aligned(arguments.spread, studies)
    logger.info(f"{len(studies)} studies shared by {len(arguments.members)} members")
    for line in AlignmentDamage().report(predictions, truth, spread, arguments.resamples):
        logger.info(line)


if __name__ == "__main__":
    main()
