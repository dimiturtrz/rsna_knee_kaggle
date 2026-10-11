"""Which diversity statistic actually predicts ensemble gain HERE — the ruler this project selects on, checked.

`python -m kneemri.evaluation.fault_ruler`

WHY THIS EXISTS. Every seat argument this project has made rests on ONE statistic: `MemberDistance`'s
rank agreement, the twin detector, and its own docstring already says it "is UNVALIDATED as a predictor
and must be read that way". On 2026-09-04 a census over 166 members was filed on that statistic, and a
literature answer came back the same hour saying it is the wrong one: Pokhrel et al. (Frontiers Digit.
Health 2026) tested five diversity metrics over 1,140 ensemble combinations and, after FDR correction,
only DOUBLE-FAULT — how often two members are wrong on the SAME case — predicted ensemble accuracy.
Disagreement rate, Q-statistic, KL divergence and CKA did not. Theisen et al. (NeurIPS 2023) give the
theory: ensemble improvement is linear in the DISAGREEMENT-ERROR RATIO, and DER < 1 predicts no gain.

A citation is not a measurement of THIS pool, and adopting one because it is published is the same
mistake as keeping the incumbent because it is ours. So this module runs Pokhrel's experiment on our
own members: it computes four candidate statistics for member pairs, measures what each pair's ensemble
ACTUALLY gains offline, and reports which statistic ranks the gains. The outcome decides the ruler, and
either outcome is a finding — if agreement wins here, the citation does not transfer and the census
stands as filed.

ERRORS ARE DEFINED IN THE SPACE THE DECISION IS MADE IN, WHICH IS WHY THERE IS NO THRESHOLD. Double-fault
is classically a count over misclassified samples, and this project has no classifier: the metric is the
mean of twelve ROC AUCs and `scores.py` says in its own header that it holds no threshold anywhere,
because AUC is invariant under any increasing map and a cut could only add a bug. But AUC IS an error
rate — it is the probability that a positive outranks a negative — so the natural unit of "wrong" here
is a discordant (positive, negative) STUDY PAIR, and every statistic below is an average over those.
That definition needs no cut, no calibration and no magic number, and it comes with a free oracle: a
member's mean error over the sampled pairs must reproduce `1 - AUC`, which `report` asserts rather than
assumes.

WHAT IT CANNOT SAY. The gains measured here are OFFLINE gains on 909 held-out studies under this pool's
vote rule, and `record_mix` has already measured that the holdout prices diversity about 2.4x too
cheaply against the board. So this ranks statistics; it does not price a seat, and it licenses no slot.
It also reads only PAIRS — the champion is six members and a pair is not a roster — for the reason
`MemberDistance` gives: a seat is lost to one redundant neighbour, and a pair is the smallest unit on
which the four statistics differ at all.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.inference.vote import RankAverage

logger = logging.getLogger(__name__)

FAULT_SEED = 20260904
"""The seed every sample here is drawn under.

Seeded because the unit gate refuses unseeded randomness, and for the reason `paired_interval` gives:
a number that moves between runs cannot settle an argument, and this one is meant to settle which ruler
the project selects members on.
"""

PAIRS_PER_LABEL = 4000
"""How many (positive, negative) study pairs are drawn per finding.

Twelve labels puts 48,000 trials behind every statistic, so a proportion's standard error is at most
0.0023. The differences these statistics have to resolve are of order 0.01 — the spread of double-fault
across our pool — so the sampling error sits about four times under the quantity being read. Drawing
every pair instead is possible on 909 studies and pointless: it would fix the fourth decimal of a number
whose third decimal is what the ranking turns on.
"""

SAMPLED_PAIRS = 1200
"""How many MEMBER pairs get their ensemble actually scored.

The four statistics come from one matrix multiply over all pairs at once; the measured gain does not —
each pair costs a rank-average and twelve AUCs. 1,200 pairs is a rank correlation whose own 95% band is
about +/-0.06, which separates "predicts" from "does not" without spending an hour on a fifth decimal.
"""

PAIRED = 2
"""How many members make the unit this module reads. Named so the subset that is scored cannot drift."""

PREDICTIVE = 0.2
"""The rank correlation below which a statistic is reported as not predicting gain in this pool.

Not a significance test and not borrowed from the citation. It is the correlation at which a statistic
explains under 4% of the variance in offline gain, which is the point past which selecting on it is
indistinguishable from selecting on the members' names. Every statistic's number is printed whatever it
reads, so the cut orders the verdict line and never hides a column.
"""


@dataclass(frozen=True)
class PairStatistics:
    """The four candidate rulers for one member pair, plus what its ensemble actually gained.

    One type rather than five parallel lists for `Pool`'s reason: the arrays are only meaningful
    together, and a statistic list reordered against a gain list produces a correlation for a pairing
    that never existed and fails nothing.
    """

    agreement: float
    """Rank agreement — the INCUMBENT ruler, `MemberDistance`'s twin detector. High means twins."""

    disagreement: float
    """How often exactly one of the two is wrong on a study pair. The metric Pokhrel found does NOT predict."""

    double_fault: float
    """How often BOTH are wrong on the same study pair. The metric Pokhrel found DOES predict. Low is better."""

    der: float
    """Disagreement over mean error rate — Theisen's ratio. Above 1 predicts gain, below 1 predicts none."""

    gain: float
    """What the pair's ensemble scored, less the better member alone. The quantity every ruler is judged against."""

    better: float
    """The stronger member's own score. A CONTROL, never a candidate ruler — see `FaultRuler.partial`."""

    spread: float
    """How far apart the two members' solo scores are. The other control, and the one that does the damage."""


class FaultRuler:
    """Every member's errors as discordant study pairs, and which pairwise statistic ranks ensemble gain."""

    def __init__(self, pool: Pool, scores: LabelScores, seed: int = FAULT_SEED) -> None:
        self.pool = pool
        self.scores = scores
        self.seed = seed

    @staticmethod
    def rank_correlation(left: Float[NDArray[np.float64], " n"], right: Float[NDArray[np.float64], " n"]) -> float:
        """Spearman correlation, by midranks — how well one ordering reproduces another.

        Midranks rather than ordinal ranks for the reason `LabelScores.area_under_curve` records: ties
        settled by array position turn a constant column into a perfect or perfectly inverted predictor.
        Written here rather than taken from a library because the project declares no scipy dependency
        and the deptry gate refuses an undeclared import.
        """
        ranked = np.array([FaultRuler.midranks(values) for values in (left, right)])
        centred = ranked - ranked.mean(axis=1, keepdims=True)
        spread = np.sqrt((centred**2).sum(axis=1))
        if not spread.all():
            return float("nan")
        return float((centred[0] * centred[1]).sum() / (spread[0] * spread[1]))

    @staticmethod
    def midranks(values: Float[NDArray[np.float64], " n"]) -> Float[NDArray[np.float64], " n"]:
        """The 1-based midrank of each value, ties sharing the mean of the ordinal ranks they span."""
        order = np.argsort(values, kind="mergesort")
        ranked = np.empty(len(values), dtype=np.float64)
        _, starts, counts = np.unique(values[order], return_index=True, return_counts=True)
        for start, count in zip(starts, counts, strict=True):
            ranked[order[start : start + count]] = start + (count + 1) / 2
        return ranked

    def errors(self) -> Float[NDArray[np.float64], "members trials"]:
        """Each member's discordance on a common sample of (positive, negative) study pairs.

        The SAME study pairs for every member, which is the whole point: double-fault is a joint over the
        two members' errors, and it can only be counted where both were asked the same question. A label
        with one class absent is skipped exactly as `area_under_curve` returns nan for it, so a rare
        finding cannot enter the sample as a column of free agreement.

        Ties contribute a half, which makes the row mean equal `1 - AUC` by construction — the oracle
        `report` checks. Raw member outputs effectively never tie; the half is there because a pool built
        from rank-averaged arrays would tie constantly, and a statistic that silently scored a tie as
        correct would read ensembles as less faulty than they are.
        """
        stacked = np.stack(self.pool.predictions)
        rng = np.random.default_rng(self.seed)
        columns = []
        for column in range(len(TARGETS)):
            positive = np.flatnonzero(self.pool.truth[:, column] > MIDPOINT)
            negative = np.flatnonzero(self.pool.truth[:, column] <= MIDPOINT)
            if not len(positive) or not len(negative):
                continue
            held = stacked[:, rng.choice(positive, PAIRS_PER_LABEL), column]
            against = stacked[:, rng.choice(negative, PAIRS_PER_LABEL), column]
            columns.append((held < against).astype(np.float64) + (held == against) / PAIRED)
        return np.concatenate(columns, axis=1)

    def rates(self, errors: Float[NDArray[np.float64], "members trials"]) -> Float[NDArray[np.float64], " members"]:
        """Each member's own error rate over the sample — its `1 - AUC`, and the denominator DER divides by."""
        return errors.mean(axis=1)

    def sampled(self) -> list[tuple[int, int]]:
        """Which member pairs get their ensemble scored, drawn without replacement from all of them.

        A sample rather than the full triangle because 166 members make 13,695 pairs and each one costs
        twelve AUCs; and drawn UNIFORMLY rather than from the pool's head, because a correlation read on
        the strongest members only would be measured inside the monoculture this module exists to look
        outside of.
        """
        rng = np.random.default_rng(self.seed + 1)
        every = [(left, right) for left in range(len(self.pool)) for right in range(left + 1, len(self.pool))]
        drawn = rng.choice(len(every), min(SAMPLED_PAIRS, len(every)), replace=False)
        return [every[index] for index in drawn]

    def statistics(self) -> list[PairStatistics]:
        """The four statistics and the measured gain, for every sampled pair.

        Double-fault and disagreement come from one matrix product over the whole error matrix rather
        than pair by pair, because the joint of two indicator rows IS their inner product over trials and
        166 rows of 48,000 is a trivial multiply. The gain cannot be batched that way — it is a vote and
        twelve AUCs — which is why the pairs are sampled and the statistics are not.
        """
        errors = self.errors()
        rates = self.rates(errors)
        both = errors @ errors.T / errors.shape[1]
        solo = [self.scores.mean(member, self.pool.truth) for member in self.pool.predictions]
        rows = []
        for left, right in self.sampled():
            fault = float(both[left, right])
            apart = float(rates[left] + rates[right] - PAIRED * fault)
            mean_rate = float(rates[left] + rates[right]) / PAIRED
            rows.append(
                PairStatistics(
                    agreement=RankAverage.agreement(self.pool.predictions[left], self.pool.predictions[right]),
                    disagreement=apart,
                    double_fault=fault,
                    der=apart / mean_rate if mean_rate else float("nan"),
                    gain=self.scores.mean(self.pool.combined((left, right)), self.pool.truth)
                    - max(solo[left], solo[right]),
                    better=max(solo[left], solo[right]),
                    spread=abs(solo[left] - solo[right]),
                )
            )
        return rows

    def partial(
        self,
        values: Float[NDArray[np.float64], " pairs"],
        gains: Float[NDArray[np.float64], " pairs"],
        controls: Float[NDArray[np.float64], "pairs controls"],
    ) -> float:
        """The same rank correlation with the QUALITY of the pair regressed out of both sides.

        THE RAW NUMBER CANNOT ANSWER THE QUESTION AND THE FIRST RUN OF THIS MODULE PROVED IT: all four
        statistics — including the two the literature says point opposite ways — correlated NEGATIVELY
        with gain, at -0.15 to -0.59. That is not four rulers failing. It is one confound, and this
        project had already measured it: `pool_census` shows the distant members of this pool are the
        WEAK ones. Gain is scored against the BETTER member, so pairing a strong member with a distant
        weak one drops the mix below the strong member alone and books a large negative gain. Every
        distance statistic then reads as a predictor of "one of these two is bad", which is a fact about
        the pool's shape and not about diversity.

        So both sides are residualised against the pair's quality — the better member's score and the
        gap between the two — in rank space, and what is left is the question actually asked: among
        pairs of COMPARABLE quality, does disagreeing predict gaining? Least squares on midranks rather
        than on the raw values, so the control is removed on the same scale the correlation is read on.
        """
        ranked = np.column_stack([self.midranks(column) for column in controls.T])
        design = np.column_stack([np.ones(len(gains)), ranked])
        residuals = [
            (self.midranks(side) - design @ np.linalg.lstsq(design, self.midranks(side), rcond=None)[0]).astype(
                np.float64
            )
            for side in (values, gains)
        ]
        return self.rank_correlation(residuals[0], residuals[1])

    def oracle(self, errors: Float[NDArray[np.float64], "members trials"]) -> str:
        """The check that the error definition is the metric's own: every member's sampled rate is its `1 - AUC`.

        Stated as a line in the report rather than kept as a private assertion, because it is the one
        number that says whether anything below may be read at all. A discordance sample that does not
        reproduce the score is a sample drawn against the wrong truth, and every correlation under it
        would be a correlation between two bugs.
        """
        measured = self.rates(errors)
        scored = np.array([1 - self.scores.mean(member, self.pool.truth) for member in self.pool.predictions])
        return (
            f"  oracle: sampled error rate against 1 - AUC over {len(measured)} members, "
            f"worst gap {float(np.abs(measured - scored).max()):.4f} (sampling error alone is ~0.0023)"
        )

    def report(self) -> list[str]:
        """Which statistic ranks offline ensemble gain in this pool, with the incumbent ruler in the table."""
        errors = self.errors()
        rows = self.statistics()
        gains = np.array([row.gain for row in rows])
        ranked = self._ranked(rows, gains)
        return [
            f"ruler check over {len(self.pool)} members, {len(rows)} sampled pairs, "
            f"{errors.shape[1]} discordance trials each, voting {self.pool.rule()}",
            self.oracle(errors),
            "  each statistic signed so that HIGHER predicts MORE gain. PARTIAL is the reading — it holds the",
            "  pair's quality fixed; RAW is the same number confounded by the weak-and-distant shape of the pool:",
            *(f"    {name}: partial {value:+.4f}   raw {plain:+.4f}" for value, plain, name in ranked),
            f"  measured gain over the pairs: median {float(np.median(gains)):+.4f}, "
            f"best {float(gains.max()):+.4f}, worst {float(gains.min()):+.4f}",
            f"  DER above 1 in {int((np.array([row.der for row in rows]) > 1).sum())} of {len(rows)} pairs",
            self._verdict(ranked[0][0], ranked[0][2]),
        ]

    def _ranked(
        self, rows: list[PairStatistics], gains: Float[NDArray[np.float64], " pairs"]
    ) -> list[tuple[float, float, str]]:
        """Every candidate ruler as `(partial, raw, name)`, best partial first, ties broken by raw then name.

        Each is signed so that HIGHER predicts MORE gain — agreement and double-fault are negated —
        and the controls are the pair's quality, which is what `partial` holds fixed.
        """
        candidates = {
            "agreement (INCUMBENT, MemberDistance)": np.array([-row.agreement for row in rows]),
            "disagreement rate": np.array([row.disagreement for row in rows]),
            "double-fault rate (negated)": np.array([-row.double_fault for row in rows]),
            "disagreement-error ratio (DER)": np.array([row.der for row in rows]),
        }
        controls = np.column_stack([[row.better for row in rows], [row.spread for row in rows]])
        return sorted(
            (
                (self.partial(values, gains, controls), self.rank_correlation(values, gains), name)
                for name, values in candidates.items()
            ),
            reverse=True,
        )

    @staticmethod
    def _verdict(best: float, winner: str) -> str:
        """The line that names the winning ruler, or says none clears `PREDICTIVE` in this pool."""
        if best >= PREDICTIVE:
            return f"  VERDICT: {winner} ranks gain best at {best:+.4f}"
        return (
            f"  VERDICT: NO statistic reaches {PREDICTIVE} here — the best is {winner} at {best:+.4f}, "
            f"so pair diversity does not predict offline gain in this pool and selecting on any of "
            f"these four is selecting on noise"
        )


def main() -> None:
    """Check the diversity ruler this project selects seats on against the gains it is supposed to predict."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument(
        "--records", type=Path, nargs="*", default=None, help="members to check; every one with a record by default"
    )
    arguments = command.parsed()
    models = command.rooted(arguments).models()
    records = arguments.records if arguments.records else sorted(models.glob("*.pt"))
    mixed = RecordMix(records)
    logger.info("checking the ruler over %d checkpoints", len(records))
    for dropped in mixed.narrow():
        logger.info("%s", dropped)
    command.report(FaultRuler(mixed.per_member(), LabelScores()).report())


if __name__ == "__main__":
    main()
