"""Is our offline ORDERING resolvable by 909 studies at all — `python -m kneemri.evaluation.rank_stability`.

THE QUESTION, AND WHY NO EXISTING TOOL ASKS IT. Every bootstrap in this repo resamples the studies to put
an interval on ONE arm's own number (`StudyBootstrap.interval`, `scores.mean_interval`). None of them
resamples the studies and re-derives the ORDER OF THE ARMS, which is the quantity every keep/kill in the
ledger actually consumed. A per-arm interval says how precisely we know a member's score; it does not say
whether the member ranked above it is really above it, and those are different questions the moment two
arms sit closer together than either interval is wide — which is the whole 336/12 family.

THE PROVENANCE. Researcher D281 §3(1), transplanting the method of Maier-Hein et al., "Is the winner
really the best?", Nat Commun 2018 (arXiv:1806.02051), Figs. 3-4. Their measurement on 56 biomedical
challenges, unchanged submissions, resampling only the test set: the first rank is stable for 9-21% of
tasks depending on the metric, a median of 15% of the OTHER teams rank first in at least 1% of bootstrap
partitions, and in one task removing a SINGLE test case put 67% of the other teams in first place. Those
percentages are theirs, from segmentation tasks scored with DSC/HD, and must not be quoted as ours — the
mechanism transfers, the numbers do not. This module is what produces ours.

WHAT IT DECIDES, stated before it was run. If the champion holds rank 1 in a small share of bootstraps,
then the ~0.010 band this project has been reading orderings out of is ONE number with error bars: every
keep/kill inside it was a coin flip, and the seven consecutive board flats are the expected reading of a
CORRECT ruler on arms separated by less than the holdout's own resampling noise — no appeal to the test
set being different is needed to explain them. If the champion holds rank 1 in most bootstraps, our
ordering is real and the board's flatness needs the test set to explain it. Those two worlds have been
argued about for a week without a measurement; this is the measurement, and it costs no card and no slot.

THE TWO SMALLER READINGS IT CARRIES, because they run off the same pool and answering them separately
would mean loading the same records three times:

  * DEFINED COLUMNS (D281 §1.1). `LabelScores.mean` is a `nanmean`, and `area_under_curve` returns NaN
    for a column with one class absent. So a column that is all-negative on the shared studies is
    silently DROPPED and our macro is a mean over fewer than twelve, while the competition's is always
    over twelve. That is a DENOMINATOR difference, not an averaging one, and nobody had counted it.
  * CUT AGREEMENT (D281 §3(2)). The recut to `ledger75` moves the LEVEL by +0.026 (knee-f9aao). Nobody
    measured whether it moves the ORDER. Same predictions, two truths — Maier-Hein Fig. 2(a) with our own
    two rulers. Tau near 1 says our truth-definition is an offset and the +0.026 may be added with a
    clear conscience; a low tau says our own two rulers disagree about which arm is better, which is
    local evidence for the inversion reading measured on us rather than borrowed.

WHAT IT IS NOT. It resamples STUDIES, so it prices the holdout's resolving power and nothing else. It is
silent about whether the competition's test set is drawn from our distribution — that is covariate shift,
it needs the test images, and D281 §3(4) names adversarial validation as the tool for it. A ruler that
cannot resolve our own arms on our own studies is a defect independent of any shift, which is why this
one runs first.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from core import cli
from core.bootstrap import STUDY_RESAMPLES, StudyBootstrap
from kneemri.evaluation.ledger75 import MIDPOINT_COLUMNS, RECUT
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.runs.holdout_record import SUFFIX

logger = logging.getLogger(__name__)

TARGET_COLUMNS = 12
"""What the competition averages over. Our `nanmean` may average over fewer; that gap is the reading."""

RESOLVED_SHARE = 0.90
"""Where a first-rank share stops being a coin flip and starts being an ordering. Legislated, not fitted:
below it the arm that ranks first on the full holdout loses that rank to a resample more than one time in
ten, which is more often than this project has ever been willing to be wrong about a seat."""


@dataclass(frozen=True)
class ArmStability:
    """One arm's rank under resampling — the share is the reading, the score is only its label."""

    name: str
    score: float
    rank: int
    first_share: float
    mean_rank: float
    defined: int

    def resolved(self) -> bool:
        """Does this arm hold the top rank often enough to call the ordering real?"""
        return self.rank == 1 and self.first_share >= RESOLVED_SHARE

    def thin(self) -> bool:
        """Is this arm's macro a mean over fewer than the twelve columns the board averages?"""
        return self.defined < TARGET_COLUMNS


class RankStability:
    """The ordering of several arms, resampled — how much of it survives the studies being redrawn."""

    def __init__(self, pool: Pool, resamples: int = STUDY_RESAMPLES) -> None:
        self.pool = pool
        self.resamples = resamples

    def ruler(self, *, recut: bool = False) -> LabelScores:
        """The midpoint every ledger number was read at, or `ledger75` — the two truths §3(2) compares."""
        if recut:
            return LabelScores(RECUT, dict.fromkeys(MIDPOINT_COLUMNS, MIDPOINT))
        return LabelScores(MIDPOINT)

    def scored(self, scores: LabelScores, drawn: NDArray[np.int64] | None = None) -> list[float]:
        """Every arm's macro on these studies, in pool order — the vector a ranking is taken over."""
        truth = self.pool.truth if drawn is None else self.pool.truth[drawn]
        return [scores.mean(member if drawn is None else member[drawn], truth) for member in self.pool.predictions]

    def defined(self, scores: LabelScores) -> list[int]:
        """How many of the twelve columns each arm's macro actually averaged over (§1.1)."""
        return [
            sum(1 for value in scores.per_label(member, self.pool.truth).values() if not np.isnan(value))
            for member in self.pool.predictions
        ]

    @staticmethod
    def order(values: list[float]) -> NDArray[np.int64]:
        """Ranks, 1 = best. NaN sinks to last so a degenerate resample cannot win a bootstrap."""
        filled = np.where(np.isnan(values), -np.inf, values)
        return np.asarray(stats.rankdata(-filled, method="average"), dtype=np.int64)

    def table(self, scores: LabelScores) -> list[ArmStability]:
        """Each arm's full-holdout rank beside the share of resamples in which it ranked FIRST.

        The share is the number D281 asks for and the mean rank is what makes it readable: an arm can
        take first place in a tenth of the draws and still sit fifth on average, which is a different
        state from one that is second in every draw.
        """
        full = self.scored(scores)
        ranks = self.order(full)
        firsts = np.zeros(len(self.pool), dtype=np.int64)
        totals = np.zeros(len(self.pool), dtype=np.float64)
        draws = 0
        for drawn in StudyBootstrap(self.resamples).draws(self.pool.truth.shape[0]):
            drawn_ranks = self.order(self.scored(scores, drawn))
            firsts += (drawn_ranks == 1).astype(np.int64)
            totals += drawn_ranks
            draws += 1
        counted = np.asarray(self.defined(scores), dtype=np.int64)
        return [
            ArmStability(
                name=self.pool.names[index],
                score=full[index],
                rank=int(ranks[index]),
                first_share=float(firsts[index]) / draws,
                mean_rank=float(totals[index]) / draws,
                defined=int(counted[index]),
            )
            for index in range(len(self.pool))
        ]

    def cut_agreement(self) -> float:
        """Kendall tau between the midpoint ordering and the `ledger75` ordering — D281 §3(2).

        The two rulers score the SAME predictions against two truths, so any disagreement here is a
        property of our label definition alone. This is the cheapest probe the project has of whether a
        truth change is an offset or an inversion for our own arms.
        """
        midpoint = self.scored(self.ruler())
        recut = self.scored(self.ruler(recut=True))
        return float(stats.kendalltau(midpoint, recut).statistic)

    def report(self, scores: LabelScores) -> list[str]:
        """The table, the cut agreement, and the verdict the first-rank share licenses."""
        rows = sorted(self.table(scores), key=lambda arm: arm.rank)
        lines = [
            f"{len(self.pool)} arm(s) over {self.pool.truth.shape[0]} shared studies, "
            f"{self.resamples} resamples, ruler {scores.name()}, vote {self.pool.rule()}",
            f"{'arm':44s} {'score':>8s} {'rank':>5s} {'first%':>7s} {'mean rk':>8s} {'cols':>5s}",
        ]
        lines += [
            f"{row.name[:44]:44s} {row.score:8.4f} {row.rank:5d} "
            f"{row.first_share * 100:6.1f}% {row.mean_rank:8.2f} {row.defined:5d}"
            for row in rows
        ]
        lines += self._verdict(rows)
        lines.append(f"cut agreement midpoint vs ledger75: Kendall tau {self.cut_agreement():+.4f}")
        return lines

    @staticmethod
    def _verdict(rows: list[ArmStability]) -> list[str]:
        """What the top arm's share says about every keep/kill taken inside this band."""
        best = rows[0]
        thin = [row.name for row in rows if row.thin()]
        lines = [
            f"top arm {best.name} holds rank 1 in {best.first_share * 100:.1f}% of resamples"
            + (
                " — the ordering is RESOLVED by this holdout"
                if best.resolved()
                else f" — below {RESOLVED_SHARE * 100:.0f}%, so this band is ONE number with error bars and"
                " a keep/kill taken inside it was a coin flip"
            ),
            f"{sum(1 for row in rows if row.first_share > 0)} of {len(rows)} arm(s) rank first "
            "in at least one resample",
        ]
        if thin:
            lines.append(
                f"THIN MACRO: {', '.join(thin)} average over fewer than {TARGET_COLUMNS} columns — "
                "a denominator difference against the board, not an averaging one"
            )
        return lines


def main() -> None:
    """Resample the studies and re-derive the ordering of the arms named on the command line."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--holdouts", type=Path, required=True, help="directory of .holdout.npz records")
    command.parser.add_argument("--arms", nargs="+", required=True, help="member stems to rank against each other")
    command.parser.add_argument("--resamples", type=int, default=STUDY_RESAMPLES)
    command.parser.add_argument("--ruler", choices=["midpoint", "ledger75"], default="midpoint")
    arguments = command.parsed()

    pool = RecordMix([arguments.holdouts / f"{name}{SUFFIX}" for name in arguments.arms]).per_member()
    stability = RankStability(pool, arguments.resamples)
    command.report(stability.report(stability.ruler(recut=arguments.ruler == "ledger75")))


if __name__ == "__main__":
    main()
