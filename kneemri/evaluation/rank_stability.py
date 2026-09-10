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

import csv
import itertools
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

CONTROL_SPLITS = 20
"""How many random halves price a statistic's spread. Twenty because each is two full passes over the
pool and the reading is a min/median/max, not an interval that would want hundreds."""

PERCENTILE_LOW = 2.5
"""The lower end of a 95% percentile interval — the same two ends every other interval here reports."""

PERCENTILE_HIGH = 97.5
"""The upper end. Named because a bare 2.5/97.5 at a call site says nothing about which interval it is."""

CONTROL_SEED = 2026
"""The project seed, so the random-split floor a reading is compared against is the same one twice."""

RESOLVED_SHARE = 0.90
"""Where a first-rank share stops being a coin flip and starts being an ordering. Legislated, not fitted:
below it the arm that ranks first on the full holdout loses that rank to a resample more than one time in
ten, which is more often than this project has ever been willing to be wrong about a seat."""


@dataclass(frozen=True)
class SeriesAxes:
    """What was ACQUIRED for each study — the only case-mix axes this dataset carries for free.

    `train_series.csv` names four candidate axes and two of them are not axes at all: `Fat_Suppression`
    and `Fluid_Sensitive` are byte-identical across all 24,371 series (one variable wearing two column
    names, so treating them as two would double-count), and every one of the 4,407 studies carries
    Axial+Coronal+Sagittal, so plane-set is constant. That leaves how MANY series a study has (3-14,
    mode 5) and what SHARE of them are fat-suppressed (0.23-0.45+), which is what this carries.

    Neither is a label, which is the point: partitioning the holdout on an acquisition fact and
    re-deriving the ordering inside each part asks whether our ranking is case-mix dependent without
    consuming a single test label.
    """

    counts: dict[str, int]
    suppressed: dict[str, float]

    @classmethod
    def read(cls, table: Path) -> "SeriesAxes":
        """Both axes off the series table, one pass, keyed by study."""
        counts: dict[str, int] = {}
        suppressed: dict[str, float] = {}
        with table.open(newline="") as handle:
            for row in csv.DictReader(handle):
                study = row["StudyInstanceUID"]
                counts[study] = counts.get(study, 0) + 1
                suppressed[study] = suppressed.get(study, 0.0) + float(row["Fat_Suppression"])
        return cls(counts, {study: total / counts[study] for study, total in suppressed.items()})

    def groups(self, studies: list[str]) -> dict[str, NDArray[np.bool_]]:
        """Four masks over these studies in row order — each axis split at its own median.

        A median split rather than terciles because the ordering is re-derived INSIDE each part and a
        thin part re-prices thirteen arms on too few studies to say anything; halves are the largest
        parts an axis can give. A study missing from the table would silently join the low side of both,
        so it is refused instead — the join is the thing most likely to be wrong here.
        """
        missing = [study for study in studies if study not in self.counts]
        if missing:
            message = f"{len(missing)} holdout studies absent from the series table, e.g. {missing[0]}"
            raise ValueError(message)
        count = np.asarray([self.counts[study] for study in studies], dtype=np.float64)
        share = np.asarray([self.suppressed[study] for study in studies], dtype=np.float64)
        return {
            "series<=median": count <= np.median(count),
            "series>median": count > np.median(count),
            "suppression<=median": share <= np.median(share),
            "suppression>median": share > np.median(share),
        }


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

    def within(self, scores: LabelScores, keep: NDArray[np.bool_]) -> list[float]:
        """Every arm's macro over a SUBSET of the studies, in pool order — one subgroup's score vector."""
        return RankStability(self.pool.restricted(keep), self.resamples).scored(scores)

    def control(self, seed: int = CONTROL_SEED) -> dict[str, NDArray[np.bool_]]:
        """Two RANDOM halves of the same studies — what a subgroup tau reads when nothing is different.

        Without this the subgroup tau is uninterpretable. Thirteen arms spanning 0.012 re-ranked on
        half the studies will disagree a great deal from sampling alone, so a low tau across an
        acquisition axis is only evidence of case-mix dependence if it is LOWER than the tau two
        meaningless halves produce. This is that floor, and it costs one more pass over the same pool.
        """
        drawn = np.random.default_rng(seed).permutation(self.pool.truth.shape[0])
        first = np.zeros(self.pool.truth.shape[0], dtype=np.bool_)
        first[drawn[: self.pool.truth.shape[0] // 2]] = True
        return {"random half A": first, "random half B": ~first}

    def subgroup_agreement(self, scores: LabelScores, groups: dict[str, NDArray[np.bool_]]) -> list[str]:
        """Does the arm ordering survive a case-mix change INSIDE our own holdout — D281 §3(3).

        The spectrum effect (Mulherin & Miller 2002) is a shift in the difficulty distribution WITHIN
        each class, and D281 §2.2's rule makes it an INVERSION route precisely when it is
        model-dependent. That is measurable here without a single test label: partition the 909 on an
        axis that is not the label, re-derive the ordering inside each part, and read the pairwise
        Kendall tau. Agreement means a case-mix shift on the competition's test set is unlikely to
        re-draw our ordering; disagreement is the inversion mechanism firing inside our own data, which
        is much stronger evidence than any borrowed table.
        """
        vectors = {name: self.within(scores, keep) for name, keep in groups.items()}
        lines = [f"{len(groups)} subgroup(s) over {self.pool.truth.shape[0]} studies, ruler {scores.name()}"]
        for name, keep in groups.items():
            best = self.pool.names[int(np.nanargmax(vectors[name]))]
            lines.append(f"{name:28s} n={int(keep.sum()):5d}  best {best[:36]:36s} {np.nanmax(vectors[name]):.4f}")
        names = list(groups)
        for first, second in itertools.combinations(names, 2):
            tau = float(stats.kendalltau(vectors[first], vectors[second]).statistic)
            lines.append(f"tau {first} vs {second}: {tau:+.4f}")
        return lines

    def subgroup_margin(self, scores: LabelScores, groups: dict[str, NDArray[np.bool_]]) -> list[str]:
        """What the ENSEMBLE buys over its best single member, per subgroup — the diversity bet, priced.

        Klontzas et al. 2024 (Eur Radiol, hip AVN MRI) is the only published number that prices the
        argument this project has been making for weeks: their cross-architecture ensemble beat the best
        single model by +0.024 internally and by **+0.066 externally** — the margin GREW 2.75x out of
        distribution. If that mechanism is real it is worth far more than any keep/kill inside our
        0.012 band, because it is the one lever sized above the ruler's own noise floor.

        Their axis is internal-vs-external and we cannot run it. The honest local proxy is the same one
        §3(3) uses: our own HARD subgroup versus our own EASY one. A margin that grows where the single
        model degrades is the mechanism firing on us; a flat margin says the Klontzas number is about
        their two hospitals and does not transfer, and the diversity bet has to be argued some other way.
        """
        lines = [f"ensemble margin over best single, {len(self.pool)} members, ruler {scores.name()}"]
        for name, keep in groups.items():
            restricted = self.pool.restricted(keep)
            singles = [scores.mean(member, restricted.truth) for member in restricted.predictions]
            together = scores.mean(restricted.combined(tuple(range(len(restricted)))), restricted.truth)
            best = float(np.nanmax(singles))
            lines.append(
                f"{name:28s} n={int(keep.sum()):5d}  best single {best:.4f}  "
                f"ensemble {together:.4f}  margin {together - best:+.4f}"
            )
        return lines

    def margin_interval(self, scores: LabelScores) -> tuple[float, float, float]:
        """The ensemble's gain over its best single member, PAIRED — value and percentile interval.

        The split-half control answers a harsher question than this one and must not be substituted for
        it: disjoint halves cost half the studies AND throw away the pairing, so their spread is an upper
        bound on a quantity nobody reads. What a decision actually consumes is the DIFFERENCE between two
        numbers taken on the SAME studies, and resampling those studies once for both terms is what
        prices it. Same reason `table` draws one index set for all thirteen arms.

        The best single member is re-chosen inside every resample. Fixing it to the full-pool winner
        would price a different quantity — the gain over ONE named member — and would read optimistically
        whenever that member is not the best on the drawn studies.
        """
        together = self.pool.combined(tuple(range(len(self.pool))))
        drawn_margins: list[float] = []
        for drawn in StudyBootstrap(self.resamples).draws(self.pool.truth.shape[0]):
            singles = self.scored(scores, drawn)
            mixed = scores.mean(together[drawn], self.pool.truth[drawn])
            drawn_margins.append(mixed - float(np.nanmax(singles)))
        full = scores.mean(together, self.pool.truth) - float(np.nanmax(self.scored(scores)))
        low, high = np.percentile(drawn_margins, [PERCENTILE_LOW, PERCENTILE_HIGH])
        return full, float(low), float(high)

    def control_spread(self, scores: LabelScores, count: int = CONTROL_SPLITS) -> list[str]:
        """The same two statistics over MANY random halves — which of them 909 studies can actually carry.

        One random split gives a floor; several give its spread, and the spread is what says whether a
        statistic is worth reading at this n at all. Two statistics are compared over the identical
        splits: the ORDERING (Kendall tau between the halves) and the ENSEMBLE MARGIN over the best
        single member. If the margin's spread is small where the tau's is large, then the quantity this
        project has been reading — which arm outranks which — is the one thing our holdout cannot
        resolve, while what an arm CONTRIBUTES to a mix is resolvable on the same studies. That would be
        a reason to change the statistic every keep/kill is taken on, which is a bigger lever than any
        arm inside the band.
        """
        taus: list[float] = []
        margins: list[float] = []
        for offset in range(count):
            halves = self.control(CONTROL_SEED + offset)
            vectors = [self.within(scores, keep) for keep in halves.values()]
            taus.append(float(stats.kendalltau(vectors[0], vectors[1]).statistic))
            for keep in halves.values():
                restricted = self.pool.restricted(keep)
                singles = [scores.mean(member, restricted.truth) for member in restricted.predictions]
                together = scores.mean(restricted.combined(tuple(range(len(restricted)))), restricted.truth)
                margins.append(together - float(np.nanmax(singles)))
        return [
            f"{count} random splits, ruler {scores.name()} — what this holdout can carry",
            f"ordering    tau      min {min(taus):+.4f}  median {float(np.median(taus)):+.4f}  max {max(taus):+.4f}",
            f"ensemble    margin   min {min(margins):+.4f}  median {float(np.median(margins)):+.4f}  "
            f"max {max(margins):+.4f}  spread {max(margins) - min(margins):.4f}",
        ]

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
    command.parser.add_argument(
        "--series", type=Path, help="train_series.csv — adds the D281 3(3) subgroup reading over acquisition axes"
    )
    arguments = command.parsed()

    pool = RecordMix([arguments.holdouts / f"{name}{SUFFIX}" for name in arguments.arms]).per_member()
    stability = RankStability(pool, arguments.resamples)
    ruler = stability.ruler(recut=arguments.ruler == "ledger75")
    lines = stability.report(ruler)
    if arguments.series is not None:
        groups = SeriesAxes.read(arguments.series).groups(pool.studies)
        lines += stability.subgroup_agreement(ruler, groups)
        lines += stability.subgroup_agreement(ruler, stability.control())
        lines += stability.subgroup_margin(ruler, groups)
        lines += stability.subgroup_margin(ruler, stability.control())
        lines += stability.control_spread(ruler)
    value, low, high = stability.margin_interval(ruler)
    lines.append(
        f"ensemble margin over best single, PAIRED: {value:+.4f}  95% [{low:+.4f}, {high:+.4f}]  width {high - low:.4f}"
    )
    command.report(lines)


if __name__ == "__main__":
    main()
