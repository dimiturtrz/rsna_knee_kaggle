"""Does a DELTA between two arms survive changing the ruler? `python -m kneemri.evaluation.ruler_transfer`.

Every number this project reports is agreement with LLM-read report labels; the competition scores
agreement with radiologist annotations. `supervision_gap` prices the two rulers by LEVEL and finds them
close — 0.8391 against the annotations, 0.8419 against the reports, on the same 58 studies, while a
fifth of the cells disagree. That is reassuring and it is not the question. A ruler may be biased by a
constant and still rank recipes correctly; what decides whether the local ledger can pick a recipe is
whether a GAP BETWEEN TWO ARMS survives the change of ruler.

So this module scores pairs of arms twice over the same studies and compares the two deltas.

THE ARMS COST NO GPU, which is the whole reason this is measurable today. The published members were
fitted on `report_labels_v2.csv` with no annotation weighting and carry out-of-fold predictions for
every study, so the radiologist annotations are genuinely held out for them. Combining the first k of
them is a real quality axis — the ledger has twenty of them at 0.8355 and a single one far below — and
it needs nothing but arithmetic over predictions that already exist.

BOTH RULERS SEE THE SAME BOOTSTRAP DRAW. Resampling studies independently per ruler would compare two
deltas that disagree partly because they were measured on different studies, which is exactly the
confound the paired bootstrap exists to remove. One draw, four scores.

WHAT A RESULT LOOKS LIKE. A ratio near 1 means the ledger measures the right axis and only its level is
biased. A ratio far from 1, with an interval that excludes 1, means part of what this project has
"measured" is a property of the label reader. An interval spanning 1 widely means 58 studies cannot
answer it, which is a fact about the instrument worth having written down rather than assumed away.

WHAT IT IS NOT. It is not the local-to-LEADERBOARD compression of knee-7d1 (0.0107 local became 0.020
on the board). That needs the board. This is the ruler half — report versus radiologist — which is the
candidate mechanism `supervision_gap.thresholds` names from the other side: our ruler calls 45.5% of
cells positive where the radiologist calls 34.4%.
"""

import argparse
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kneemri.evaluation.scores import BOOTSTRAP_SEED, RESAMPLES, LabelScores

logger = logging.getLogger(__name__)

SIZES = (1, 2, 4)
"""The member counts to build arms from: four seeds cover every study, so four is the ceiling."""

SHARES = (0.05, 0.1, 0.2, 0.4)
"""How much of a permuted copy to blend in, chosen to bracket the 1.5% result bar from both sides."""

SEEDS = (11, 101, 2026, 7777, 31337)
"""Permutation seeds. Five, because one of them alone said something four others contradicted."""

RATIO_FLOOR = 0.002
"""A resample whose report-ruler delta is smaller than this contributes no ratio.

Dividing by a delta that has bootstrapped to nearly zero produces an arbitrarily large ratio and would
make the interval a report on the denominator rather than on the transfer.
"""


@dataclass(frozen=True)
class Transfer:
    """One arm pair scored on both rulers, with the paired interval of each delta and of their ratio."""

    reported: float
    annotated: float
    reported_interval: tuple[float, float]
    annotated_interval: tuple[float, float]
    ratio_interval: tuple[float, float]
    ratios: int

    @property
    def ratio(self) -> float:
        """How much of the report-ruler delta the annotation ruler returns, at the point estimate."""
        return self.annotated / self.reported if abs(self.reported) >= RATIO_FLOOR else float("nan")


class RulerTransfer:
    """Two arms, two rulers, one bootstrap draw shared between them."""

    def __init__(self, scores: LabelScores | None = None, resamples: int = RESAMPLES) -> None:
        self.scores = scores or LabelScores()
        self.resamples = resamples

    def arms(
        self, per_member: list[NDArray[np.float64]], sizes: tuple[int, ...] = SIZES
    ) -> dict[int, NDArray[np.float64]]:
        """The combined prediction of the first k members, for each k that exists.

        First k rather than a random subset: the members are interchangeable seeds of one recipe, so any
        subset of size k is the same arm up to noise, and a fixed prefix is the one a reader can
        recompute.
        """
        from kneemri.inference.vote import RankAverage  # noqa: PLC0415 - analysis-only; the trainer must not import it

        return {size: RankAverage().combine(per_member[:size]) for size in sizes if size <= len(per_member)}

    def weakened(self, arm: NDArray[np.float64], share: float, seed: int = BOOTSTRAP_SEED) -> NDArray[np.float64]:
        """The arm blended toward a permutation of itself — a quality drop of a size we choose.

        THE MEMBER LADDER IS TOO SHORT TO READ, which is what the first run of this module measured:
        combining one member against four moves the report ruler by +0.0043 with a band from -0.0037 to
        +0.0124, and a ratio computed off a denominator that small is a report on the denominator. The
        arms available without a GPU are four seeds of ONE recipe and they are simply not far apart.

        A ruler does not care how an arm got worse, only whether it can see that it did. Blending in a
        permuted copy degrades an arm by a controllable amount and lets the transfer be measured where it
        matters — at the 1.5% the owner's bar is set at, and above it.

        WHAT THIS DOES NOT MODEL: a real recipe change helps some findings and hurts others, while this
        degrades all twelve the same way. So it prices the ruler's SENSITIVITY, not the per-label
        structure of a genuine arm difference.
        """
        permuted = arm[np.random.default_rng(seed).permutation(arm.shape[0])]
        return (1.0 - share) * arm + share * permuted

    def transfer(
        self,
        weaker: NDArray[np.float64],
        stronger: NDArray[np.float64],
        reported: NDArray[np.float32],
        annotated: NDArray[np.float32],
    ) -> Transfer:
        """The gap between two arms, measured against both label sources on the same studies."""
        generator = np.random.default_rng(BOOTSTRAP_SEED)
        studies = annotated.shape[0]
        drawn_reported: list[float] = []
        drawn_annotated: list[float] = []
        ratios: list[float] = []
        for _ in range(self.resamples):
            drawn = generator.integers(0, studies, size=studies)
            by_report = self.gap(weaker[drawn], stronger[drawn], reported[drawn])
            by_annotation = self.gap(weaker[drawn], stronger[drawn], annotated[drawn])
            drawn_reported.append(by_report)
            drawn_annotated.append(by_annotation)
            if abs(by_report) >= RATIO_FLOOR:
                ratios.append(by_annotation / by_report)
        return Transfer(
            reported=self.gap(weaker, stronger, reported),
            annotated=self.gap(weaker, stronger, annotated),
            reported_interval=self.interval(drawn_reported),
            annotated_interval=self.interval(drawn_annotated),
            ratio_interval=self.interval(ratios),
            ratios=len(ratios),
        )

    def gap(
        self,
        weaker: NDArray[np.float64],
        stronger: NDArray[np.float64],
        targets: NDArray[np.float32],
    ) -> float:
        """The mean-over-findings difference between two arms against one label source."""
        return self.scores.mean(stronger, targets) - self.scores.mean(weaker, targets)

    def interval(self, values: list[float]) -> tuple[float, float]:
        """The 95% percentile interval, or NaNs when nothing was collected."""
        if not values:
            return (float("nan"), float("nan"))
        return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))

    def spread(
        self,
        arm: NDArray[np.float64],
        reported: NDArray[np.float32],
        annotated: NDArray[np.float32],
        shares: tuple[float, ...] = SHARES,
        seeds: tuple[int, ...] = SEEDS,
    ) -> list[str]:
        """The ratio at each degradation size, ACROSS permutation seeds rather than within one.

        THIS EXISTS BECAUSE ONE SEED LIED. At the default seed, share 0.40 gave report +0.0430 against
        annotation +0.0730 — a ratio of 1.70 whose bootstrap interval was [1.07, 3.48] and therefore
        excluded 1, which reads as "the local ruler compresses deltas" and would have matched the
        leaderboard compression knee-7d1 measured from two points. Four more permutation seeds put the
        same share at 0.90, 1.14, 1.21 and 0.79.

        The bootstrap interval was honest about the STUDIES and silent about the permutation, which is a
        second source of variation the same size or larger. Both have to be reported, so the headline is
        the seed range and not any one seed's band.
        """
        rows = [f"{'share':>7s} {'report':>9s} {'annot':>9s} {'ratio':>8s}   across {len(seeds)} permutation seeds"]
        for share in shares:
            ratios: list[float] = []
            reports: list[float] = []
            annotations: list[float] = []
            for seed in seeds:
                weaker = self.weakened(arm, share, seed=seed)
                by_report = self.gap(weaker, arm, reported)
                by_annotation = self.gap(weaker, arm, annotated)
                reports.append(by_report)
                annotations.append(by_annotation)
                if abs(by_report) >= RATIO_FLOOR:
                    ratios.append(by_annotation / by_report)
            band = f"[{min(ratios):+.2f},{max(ratios):+.2f}]" if ratios else "[undefined]"
            rows.append(
                f"{share:7.2f} {float(np.mean(reports)):+9.4f} {float(np.mean(annotations)):+9.4f} "
                f"{float(np.mean(ratios)) if ratios else float('nan'):+8.2f} {band}"
            )
        return rows

    def lines(self, pairs: dict[str, Transfer], studies: int) -> list[str]:
        """Every arm pair, both deltas beside each other, so the comparison is the thing on the page."""
        rows = [
            f"{studies} studies scored twice; {self.resamples} paired resamples",
            f"{'arms':>9s} {'report':>9s} {'report 95%':>18s} {'annot':>9s} {'annot 95%':>18s} {'ratio':>8s}",
        ]
        for name, result in pairs.items():
            low, high = result.ratio_interval
            rows.append(
                f"{name:>9s} {result.reported:+9.4f} "
                f"[{result.reported_interval[0]:+.4f},{result.reported_interval[1]:+.4f}] "
                f"{result.annotated:+9.4f} "
                f"[{result.annotated_interval[0]:+.4f},{result.annotated_interval[1]:+.4f}] "
                f"{result.ratio:+8.2f} [{low:+.2f},{high:+.2f}]"
            )
        return rows


def main() -> None:
    """Score a ladder of published-member arms against both label sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="the report-derived table, read raw")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument(
        "--shares", type=float, nargs="*", default=list(SHARES), help="permutation blends, for deltas worth reading"
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from core.paths import DataRoot  # noqa: PLC0415 - CLI-only
    from kneemri.evaluation.oof_oracle import OutOfFoldOracle  # noqa: PLC0415 - CLI-only
    from kneemri.evaluation.supervision_gap import SupervisionGap  # noqa: PLC0415 - CLI-only

    root = DataRoot.from_config(arguments.paths)
    gap = SupervisionGap()
    identifiers, truth = gap.annotated(root.raw() / "train.csv")
    names, per_member = OutOfFoldOracle().corpus(root.processed() / "published")

    position = {name: index for index, name in enumerate(names)}
    keep = [index for index, name in enumerate(identifiers) if name in position]
    rows = np.array([position[identifiers[index]] for index in keep])
    annotated = truth[keep]
    reported = gap.reported(arguments.labels, [identifiers[index] for index in keep])
    logger.info("%d annotated studies, %d of them with a held-out prediction", len(identifiers), len(keep))

    transfer = RulerTransfer(resamples=arguments.resamples)
    arms = transfer.arms([member[rows] for member in per_member])
    ladder = sorted(arms)
    best = arms[ladder[-1]]
    pairs = {
        f"{weaker}->{stronger}": transfer.transfer(arms[weaker], arms[stronger], reported, annotated)
        for weaker, stronger in itertools.pairwise(ladder)
    }
    pairs[f"perm {arguments.shares[-1]:.2f}"] = transfer.transfer(
        transfer.weakened(best, arguments.shares[-1]), best, reported, annotated
    )
    for line in transfer.lines(pairs, len(keep)):
        logger.info(line)
    for line in transfer.spread(best, reported, annotated, tuple(arguments.shares)):
        logger.info(line)


if __name__ == "__main__":
    main()
