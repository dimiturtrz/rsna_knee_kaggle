"""What does a UNIT of member diversity buy in ensemble margin — `python -m kneemri.evaluation.diversity_slope`.

THE QUESTION, AND WHY IT IS THE ONE LEFT. On 2026-09-10 `rank_stability.margin_interval` priced what our
whole thirteen-member ensemble buys over its best single member: **+0.0063, 95% [+0.0035, +0.0090]** — real
(the interval excludes zero) and worth a quarter of the 1.5% significance bar. Adding the only member from
a different architecture family moved it **+0.0009**, a sixth of that interval's own width. Klontzas et al.
2024 report +0.024 internal and +0.066 external for a cross-architecture ensemble, four to ten times ours,
across three architectures that differ far more than our thirteen variations of one recipe.

That is two points and no axis. The obvious reading — "our members are too correlated for the diversity
mechanism to have anything to work with" — is a story until something measures the RELATION between how
different a pool's members are and what the pool buys. This module measures it, on our own records, with
no card and no slot, and the deliverable is a SLOPE: how much margin one unit of decorrelation is worth
here. With a slope, "train a genuinely different family" stops being a hunch and becomes arithmetic — read
the diversity a new family would plausibly bring, multiply, compare against 1.5%.

THE CONFOUND, NAMED BEFORE THE NUMBER. Within one pool, diversity and margin are both entangled with
member QUALITY, and the entanglement has a sign in each direction. A subset that reads as unusually
decorrelated is often decorrelated because one of its members is BAD — and a bad member both lowers the
best-single term (inflating the margin) and adds an ordering nobody else has (raising the diversity). So a
raw diversity-margin correlation is guaranteed to come out positive whether or not diversity does anything,
and quoting it would be the same class of error as the split-half margin spread this replaces. The module
therefore reports THREE numbers and refuses to be read off the first alone:

  * the RAW slope of margin on diversity across subsets,
  * the slope of margin on the BEST single score and on the MEAN single score (the confound's own
    strength on each end, measured not assumed),
  * the PARTIAL slope — margin on diversity, both residualised on BOTH quality terms, which is the only
    one of the four that is an answer to the question asked.

Both quality terms are needed and the best-single alone is not enough: residualising on it fixes a
subset's CEILING and leaves its floor free, so a subset of one good member and five bad ones reads as
high-diversity at unchanged best-single, and its margin is NEGATIVE because the vote drags the good
member down. That is a quality reading wearing the diversity axis' clothes.

If the partial slope collapses to zero while the raw slope is healthy, the honest finding is that this
pool's apparent diversity effect IS the quality effect wearing a costume, and no amount of decorrelating
our own recipe variations will buy anything.

WHAT IT IS NOT. Every subset is drawn from ONE pool of one project's recipes, so the diversity axis it
spans is the axis our recipes happen to span. A slope fitted here extrapolates to a genuinely foreign
architecture only as far as that architecture's decorrelation lands inside the observed range — which the
report prints, so extrapolation past it is visible rather than silent. And the margin per subset is a
single point estimate, not an interval: the noise is real and large (±0.0028 on one subset) and is dealt
with by drawing many subsets, never by trusting one.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from core import cli
from kneemri.evaluation.ledger75 import MIDPOINT_COLUMNS, RECUT
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.runs.holdout_record import SUFFIX

logger = logging.getLogger(__name__)

SUBSETS = 300
"""How many random subsets carry the slope. Each is one margin whose own noise is about half the whole
effect, so the slope is only readable because many of them average it away — 300 puts the standard error
on the fit an order of magnitude under the per-subset scatter."""

SUBSET_SEED = 2026
"""The project seed, so a re-run of this fit is the same fit."""

QUALITY_QUARTILE = 0.25
"""Which slice of the subsets counts as 'good' when asking how much diversity good members reach. A
quarter of 300 draws is 75 subsets — enough that the span is not one lucky draw, narrow enough that it
is still describing the good end rather than the middle."""

MIN_SEATS = 4
"""Below four members a 'margin over the best single' is dominated by which single happened to be best."""

MAX_SEATS = 16
"""Above sixteen the count plateau (record_mix: 0.0030 across six to twenty-four seats) means extra seats
move the diversity axis and not much else, which is exactly the range this wants — but the vote also gets
slow, and the plateau's own reading says nothing past this is a different regime."""


@dataclass(frozen=True)
class SubsetReading:
    """One random subset, priced on both axes at once — a single point of the fit."""

    seats: int
    diversity: float
    margin: float
    best_single: float
    mean_single: float


class DiversitySlope:
    """Draw subsets of a member pool and fit what decorrelation is worth in margin."""

    def __init__(self, pool: Pool, scores: LabelScores, subsets: int = SUBSETS) -> None:
        self.pool = pool
        self.scores = scores
        self.subsets = subsets

    def diversity(self, subset: tuple[int, ...]) -> float:
        """One minus the mean pairwise Spearman correlation over the subset's members.

        Spearman rather than Pearson because the shipped vote is a rank average over per-label orderings
        and a monotone rescaling of a member's scores must not read as a difference. Flattened across all
        twelve columns rather than averaged per column, because a member that is redundant on eleven and
        distinct on one is redundant, and a per-column mean would let the one column speak for the member.

        Reported as one MINUS the correlation so the axis points the way the reading does: larger is more
        diverse, and a positive slope means diversity buys margin.
        """
        flattened = [self.pool.predictions[index].ravel() for index in subset]
        pairs = [
            float(stats.spearmanr(flattened[first], flattened[second]).statistic)
            for first in range(len(subset))
            for second in range(first + 1, len(subset))
        ]
        return 1.0 - float(np.mean(pairs))

    def price(self, subset: tuple[int, ...]) -> SubsetReading:
        """This subset's margin over its own best single member, beside the two terms that made it."""
        singles = [self.scores.mean(self.pool.predictions[index], self.pool.truth) for index in subset]
        best = float(np.nanmax(singles))
        together = self.scores.mean(self.pool.combined(subset), self.pool.truth)
        return SubsetReading(len(subset), self.diversity(subset), together - best, best, float(np.nanmean(singles)))

    def draw(self, seed: int = SUBSET_SEED) -> list[SubsetReading]:
        """Random subsets across the whole seat range, so the fit spans counts as well as diversity."""
        generator = np.random.default_rng(seed)
        ceiling = min(MAX_SEATS, len(self.pool))
        readings: list[SubsetReading] = []
        for _ in range(self.subsets):
            seats = int(generator.integers(MIN_SEATS, ceiling + 1))
            subset = tuple(int(index) for index in generator.choice(len(self.pool), size=seats, replace=False))
            readings.append(self.price(subset))
        return readings

    @staticmethod
    def slope(x: list[float], y: list[float]) -> tuple[float, float]:
        """Ordinary least squares slope and its two-sided p — the fit and whether it is a fit at all."""
        fit = stats.linregress(x, y)
        return float(fit.slope), float(fit.pvalue)

    @staticmethod
    def residuals(values: list[float], against: list[list[float]]) -> list[float]:
        """What is left of `values` once EVERY named confound has been regressed out of it.

        A list of covariates rather than one, because controlling for the best member does not control
        for member quality. Residualising on the best single alone fixes a subset's CEILING and leaves
        its floor free, so a subset holding one good member and five bad ones reads as high-diversity at
        unchanged best-single — and its margin is negative, because the vote drags the good member down.
        That is a quality reading wearing the diversity axis' clothes, and it is what makes the full-pool
        raw slope negative. The mean single score is the second covariate that closes it.

        The p that a caller then takes on these residuals is not adjusted for the covariates' own degrees
        of freedom, so it is very slightly anticonservative; at n=300 against two covariates that is noise
        in the fourth decimal of the p, and the slope itself is exact.
        """
        observed = np.asarray(values, dtype=np.float64)
        design = np.column_stack([np.ones(observed.shape[0]), *[np.asarray(column) for column in against]])
        coefficients, *_ = np.linalg.lstsq(design, observed, rcond=None)
        return [float(residual) for residual in observed - design @ coefficients]

    def report(self, readings: list[SubsetReading]) -> list[str]:
        """The three slopes, the range they were fitted over, and what they imply for a foreign family."""
        diversity = [reading.diversity for reading in readings]
        margin = [reading.margin for reading in readings]
        best = [reading.best_single for reading in readings]
        average = [reading.mean_single for reading in readings]
        quality = [best, average]
        raw, raw_p = self.slope(diversity, margin)
        confound, confound_p = self.slope(best, margin)
        floor, floor_p = self.slope(average, margin)
        partial, partial_p = self.slope(self.residuals(diversity, quality), self.residuals(margin, quality))
        span = max(diversity) - min(diversity)
        reachable = self.reachable_span(readings)
        lines = [
            f"{len(readings)} random subsets of {len(self.pool)} members, {MIN_SEATS}-{min(MAX_SEATS, len(self.pool))}"
            f" seats, over {self.pool.truth.shape[0]} studies, ruler {self.scores.name()}, vote {self.pool.rule()}",
            f"diversity (1 - mean pairwise spearman)  min {min(diversity):.4f}  max {max(diversity):.4f}"
            f"  span {span:.4f}",
            f"margin over best single                 min {min(margin):+.4f}  median {float(np.median(margin)):+.4f}"
            f"  max {max(margin):+.4f}",
            f"RAW      margin on diversity   slope {raw:+.4f} per unit  p {raw_p:.2g}",
            f"CONFOUND margin on best single slope {confound:+.4f} per unit  p {confound_p:.2g}",
            f"CONFOUND margin on mean single slope {floor:+.4f} per unit  p {floor_p:.2g}",
            f"PARTIAL  margin on diversity   slope {partial:+.4f} per unit  p {partial_p:.2g}   <- the answer",
        ]
        lines += [
            f"what the whole observed diversity span is worth at the partial slope: {partial * span:+.4f}"
            f" — against a 0.0150 significance bar",
            f"but the span reached by the top-{QUALITY_QUARTILE:.0%} subsets by mean member quality is only"
            f" {reachable:.4f}, worth {partial * reachable:+.4f} — everything past it is decorrelated"
            f" BECAUSE it is bad, so the partial slope out there is a prediction and not an observation",
        ]
        return lines

    @staticmethod
    def reachable_span(readings: list[SubsetReading]) -> float:
        """The diversity span among the BEST subsets — the part of the axis a good member could occupy.

        The partial slope prices diversity at fixed quality, which is the right quantity and is also a
        model: nothing in this pool is simultaneously high-quality and highly decorrelated, so the slope
        at that corner is extrapolated from subsets that are one thing or the other. This bounds how far
        the extrapolation reaches by asking what diversity the pool actually achieves WITHOUT paying for
        it in quality. A foreign family landing past this span is a bet that a good model can be that
        different — which is exactly the bet, and it should be stated rather than hidden in a slope.
        """
        floor = float(np.quantile([reading.mean_single for reading in readings], 1.0 - QUALITY_QUARTILE))
        best = [reading.diversity for reading in readings if reading.mean_single >= floor]
        return max(best) - min(best)


def main() -> None:
    """Fit what member decorrelation is worth in ensemble margin, over records already on disk."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--holdouts", type=Path, required=True, help="directory of .holdout.npz records")
    command.parser.add_argument("--arms", nargs="+", required=True, help="member stems to draw subsets from")
    command.parser.add_argument("--subsets", type=int, default=SUBSETS)
    command.parser.add_argument("--ruler", choices=["midpoint", "ledger75"], default="midpoint")
    command.parser.add_argument(
        "--largest-pool",
        action="store_true",
        help="keep the largest agreeing split instead of refusing — REQUIRED when the arms were globbed, "
        "and it names every drop, because the pool that remains is the denominator",
    )
    arguments = command.parsed()

    mix = RecordMix([arguments.holdouts / f"{name}{SUFFIX}" for name in arguments.arms])
    if arguments.largest_pool and (dropped := mix.narrow()):
        logger.info(f"narrowed to the largest agreeing split, dropping {len(dropped)}: {', '.join(dropped[:6])} …")
    pool = mix.per_member()
    ruler = (
        LabelScores(RECUT, dict.fromkeys(MIDPOINT_COLUMNS, MIDPOINT))
        if arguments.ruler == "ledger75"
        else LabelScores(MIDPOINT)
    )
    fit = DiversitySlope(pool, ruler, arguments.subsets)
    command.report(fit.report(fit.draw()))


if __name__ == "__main__":
    main()
