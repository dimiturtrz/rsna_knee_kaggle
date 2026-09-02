"""What does each additional ensemble member actually buy?

`python -m kneemri.evaluation.member_curve --checkpoints a.pt b.pt c.pt --labels labels.csv`

This question was answered once already and the answer was void. The members were trained at different
model seeds, the fold split was derived from that seed, and so each member was scored on a holdout that
was training data for the others — which reported an ensemble of five at 0.8807 when its own members
logged 0.83. The split is now a property of the corpus (`--split-seed`), so members that differ only in
initialisation share one holdout and the comparison means something.

The curve is the deciding number for two open questions at once. It says whether our fifth member earns
its forward passes at all, and the Efficiency Prize track turns entirely on that: teams scoring exactly
our 0.888 sit at efficiency ranks 29-50, while the best score on that board sits at 638, so a cheap
submission at nearly our accuracy is worth far more there than a costly one.

Every subset of each size is scored, not one arbitrary subset, because with three or five members the
spread between the best and worst subset of a given size is the honest error bar on the mean.
"""

import itertools
import logging
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from core import cli
from core.data.corpus.folds import ReportFolds
from core.data.labels.targets import Targets
from core.wakelock import Wakelock
from kneemri.device import VisibleCards
from kneemri.evaluation.label_combine import LabelCombine
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.inference.holdout_predictions import HoldoutPredictions
from kneemri.inference.member_readings import MemberReadings
from kneemri.inference.members import Member
from kneemri.inference.prediction_cache import PredictionCache
from kneemri.inference.test_time import SpatialViews
from kneemri.inference.vote import LogitAverage, QualityWeightedRank, RankAverage

logger = logging.getLogger(__name__)

COMBINE_SEED = 2026
"""Which studies fit the per-label choice and which report it — the corpus split seed, since both are
arbitrary-but-fixed partitions of the same corpus and a second free number would only be a second thing
to accidentally vary."""

HONEST_SEATS_REPORTED = 24
"""How many seats of the honest curve are printed. The greedy orders every member in the pool — 77 of
them on the current disk — and the decision this reading serves is where the count SATURATES, which the
ledger puts inside the first ten. The tail is the same members in a fixed order scoring monotonically
worse and reading it has never changed a roster."""

BOARD_RESOLVABLE = 0.0040
"""The offline seat delta below which this ruler has been MEASURED to order the board backwards.

IT IS A SPENDING GUARD, NOT A TARGET, and the distinction is the one thing to keep straight about it.
This project's bar for a RESULT is ~1.5% absolute and its noise floor is 0.0062; 0.0040 is under both. So
nothing here is a number to chase or to plan an arm around — it exists to stop a slot being spent on a
move the board cannot read, which is a question about a submission rather than about a method. An axis
worth arming has to argue for 0.015, not for this.

RAISED FROM 0.0015 ON 2026-09-02, and the old value was not merely conservative — it was refuted. It was
fitted on SIX paired entries (knee-825t) as "the plateau `longseat`'s +0.0012 tie sits above". The record
now holds 34 rows carrying both an honest offline number and a board number, which is 561 ordered pairs,
and 35 of them (6.2%) INVERT: the entry with the higher offline scored lower on the board. The largest
offline separation at which the record still inverts is 0.0037 (`gradeswap_v1` 0.8694 / board 0.933
against `peak8_v1` 0.8731 / board 0.932). Above ~0.004 the ordering is stable.

So 0.0015 was not a floor on what had been demonstrated; it sat squarely inside the band where this
ruler is known to get the SIGN wrong, and every arm killed or shipped on a delta between 0.0015 and
0.0037 was decided by a ruler that inverts there. The value is now the measured inversion ceiling rather
than a plateau read off six points.

TWO THINGS IT DOES NOT SAY. It is not a claim that a 0.005 offline gain buys a board tick — the board
quantises to 0.001 and its own sampling error is several times that (Q-BX). And an inversion of one
board quantum is one rounding step, so the 6.2% rate is an upper bound on real disagreement; what the
0.0037 figure bounds is the separation at which disagreement stops appearing at all.

It documents this constant rather than the seed above it, which is what it used to do — a copy-paste that
left the number gating a submission decision described as a corpus partition."""

BOARD_COST_MULTIPLE = 3
"""What the board charges for a negative offline seat delta, as a multiple of the offline number.

Three paired entries: -0.0013 offline became -0.004 and -0.0023 became -0.005 (knee-c0u9). The holdout
ranks correctly and errs in magnitude, and only in the direction of understating a weak member's cost —
so an offline delta inside the noise floor is not a free look, and this is the factor that says so."""

LABEL_HEADROOM = 0.90
"""The per-label AUC a headroom column lifts one finding to, so twelve holes are priced on one ruler.

Not a target and not a prediction. It is the level our BEST-scoring labels already reach on this same
holdout, which makes it the one level a reader cannot dismiss as unreachable in principle: the mix
demonstrably attains it somewhere, so a label sitting far below it is a gap in this design rather than a
statement about how hard that finding is. A common level is what makes the twelve numbers comparable —
lifting each label to its own plausible ceiling would price them against twelve different rulers."""


class MemberCurve:
    """Mean score against ensemble size, over every subset of that size."""

    def __init__(self, scores: LabelScores, combine_seed: int = COMBINE_SEED) -> None:
        self.scores = scores
        self.combine_seed = combine_seed

    def sizes(self, pool: Pool) -> dict[int, list[float]]:
        """Every subset's score, grouped by how many members it holds.

        All subsets rather than a sample: with five members that is 31 combinations and a few seconds of
        numpy, and the spread within a size is the only error bar available without retraining.
        """
        return {size: [score for score, _ in self.subsets(pool, size)] for size in range(1, len(pool) + 1)}

    def subsets(self, pool: Pool, size: int) -> list[tuple[float, tuple[int, ...]]]:
        """Every subset of one size, as (score, member indices).

        The indices are the point. `sizes` reported that five members reach 0.8601 and never said WHICH
        five, so the best number this analysis produces could not be acted on — the next entry has to
        name its members. An analysis that reports a score nobody can reproduce as a submission is doing
        half its job.
        """
        return [
            (self.scores.mean(pool.combined(subset), pool.truth), subset)
            for subset in itertools.combinations(range(len(pool)), size)
        ]

    def greedy(self, pool: Pool) -> list[str]:
        """Forward selection: the member that adds most, added, until adding one stops paying.

        `subsets` is exhaustive and therefore exponential — five members is 31 combinations and sixteen
        is 65,535 — so the pool this project has actually accumulated cannot be searched that way at
        all. Greedy is O(n^2) combines and answers the question the exhaustive search is run for: which
        members to package, and where the count saturates.

        It is a HEURISTIC and the ledger's own numbers say why that is tolerable here: the curve is
        flat once past four members, so the difference between the greedy path and the best path is
        inside the offline-to-board transfer error that already governs the decision. What it must not
        be read as is a score — see this module's caller for the two reasons.
        """
        return [
            f"greedy {seat + 1}: {score:.4f}  +{pool.names[index]}"
            for seat, (score, index) in enumerate(self.forward(pool))
        ]

    def forward(self, pool: Pool) -> list[tuple[float, int]]:
        """The greedy's path itself — each seat's score and the member that took it, in order.

        Separated from `greedy` because the ORDER is now asked for twice and the score-on-this-pool is
        only wanted once. `honest` fits this path on half the studies and scores the resulting prefixes
        on the other half, which needs the indices and must throw the fitted scores away; formatting
        them into a line first and parsing them back would be the same computation done twice, differing
        in whichever of the two copies was edited last.
        """
        chosen: list[int] = []
        path: list[tuple[float, int]] = []
        remaining = set(range(len(pool)))
        while remaining:
            score, index = max(
                (
                    (self.scores.mean(pool.combined([*chosen, candidate]), pool.truth), candidate)
                    for candidate in remaining
                ),
                key=lambda pair: pair[0],
            )
            chosen.append(index)
            remaining.discard(index)
            path.append((score, index))
        return path

    def honest(self, pool: Pool) -> list[str]:
        """The greedy's count, chosen on studies it is NOT scored on. The only seat number worth quoting.

        WHY THIS EXISTS. `greedy` searches 77 members for the prefix that maximises the holdout mean and
        then REPORTS that maximum — selection and scoring on the same 909 studies. Its own docstring
        already says the path must not be read as a score, and the caller repeats it, and the number was
        quoted as one anyway (0.8717 at eight to ten members, 2026-08-22). It is the identical mistake
        that voided this module's first answer, where members were scored on holdouts that were training
        data for the others, and the identical one that killed per-label selection — which is why the
        fitted label rule next door is reported SPLIT-HALF and is the only column its own docstring
        says is worth reading. Forward selection over a large pool is a fitting procedure; a fitting
        procedure scored in-sample reports its own overfitting as a result.

        Both directions of the split, averaged, because one half throws away half the evidence about a
        choice whose whole question is whether it survives being made elsewhere — the same argument
        `LabelCombine.halves` is built on, and it is reused here rather than re-derived.

        The gap between this and `greedy` at the same count is the SELECTION BIAS, stated rather than
        subtracted. A count whose honest score peaks well below its in-sample one is a count that was
        fitted; a plateau that survives is a plateau.
        """
        splits = LabelCombine(self.scores).halves(len(pool.truth), self.combine_seed)
        out_of_sample: dict[int, list[float]] = {}
        for fit, report in splits:
            order = [index for _, index in self.forward(pool.rows(fit))]
            scored = pool.rows(report)
            for seats in range(1, len(order) + 1):
                prefix = order[:seats]
                out_of_sample.setdefault(seats, []).append(self.scores.mean(scored.combined(prefix), scored.truth))
        held = {seats: float(np.mean(scores)) for seats, scores in out_of_sample.items()}
        best = max(held, key=lambda seats: held[seats])
        in_sample = [score for score, _ in self.forward(pool)]
        lines = [
            "honest seats: greedy refitted on half the studies, scored on the other half, both directions",
            *(
                f"  {seats:2d} members: held-out {held[seats]:.4f}   in-sample {in_sample[seats - 1]:.4f}"
                f"   bias {in_sample[seats - 1] - held[seats]:+.4f}"
                for seats in sorted(held)
                if seats <= HONEST_SEATS_REPORTED
            ),
            f"best honest count {best} at {held[best]:.4f}, against the in-sample peak "
            f"{max(in_sample):.4f} at {in_sample.index(max(in_sample)) + 1}",
        ]
        return lines

    def diverse(self, pool: Pool) -> list[str]:
        """Forward selection by DISAGREEMENT rather than by mean — the seat rule nobody has run.

        THE QUESTION IT EXISTS TO ANSWER, asked repeatedly and never tested. `greedy` picks the member
        that lifts the holdout mean most, which is the QUALITY ruler choosing seats; every architecturally
        distant member this project has trained was rejected by it, and on 2026-08-24 the distance ruler
        priced seven trunks and ordered them EXACTLY opposite to their solo scores. So the greedy's
        verdict on architecture diversity may be the verdict of the ruler rather than of the axis, and
        the only way to tell is to build the ensemble the other way and compare the two PEAKS.

        The rule: seed with the strongest member, then repeatedly take the member whose agreement with
        its nearest already-seated neighbour is lowest — the twin detector run backwards. The mean is
        computed and printed at every seat count but never consulted, so the two paths differ in the
        selection rule alone and the curves are read against each other.

        Parameter-free on purpose. A distance THRESHOLD or a quality-times-distance weighting would both
        be knobs, and a knob on a rule invented to settle an argument is how a favourable answer gets
        tuned out of a pool of 68 members. Where this loses to `greedy` is the finding, not a reason to
        add a term.
        """
        chosen = [max(range(len(pool)), key=lambda index: self.scores.mean(pool.predictions[index], pool.truth))]
        remaining = set(range(len(pool))) - set(chosen)
        lines = [f"diverse 1: {self.scores.mean(pool.combined(chosen), pool.truth):.4f}  +{pool.names[chosen[0]]}"]
        while remaining:
            index = min(
                remaining,
                key=lambda candidate: max(
                    RankAverage.agreement(pool.predictions[candidate], pool.predictions[seat]) for seat in chosen
                ),
            )
            nearest = max(RankAverage.agreement(pool.predictions[index], pool.predictions[seat]) for seat in chosen)
            chosen.append(index)
            remaining.discard(index)
            score = self.scores.mean(pool.combined(chosen), pool.truth)
            lines.append(f"diverse {len(chosen)}: {score:.4f}  +{pool.names[index]}  (nearest {nearest:.4f})")
        return lines

    def swap(self, roster: Pool, candidates: Pool) -> list[str]:
        """What each candidate is worth in the best SEAT it could take, holding the member count fixed.

        THE READING THE ROSTER POLICY ACTUALLY PERMITS, and the one nothing computed. `marginal` prices a
        candidate as an ADDITIONAL member, and `comp/champion.json`'s FOUR MEMBERS principle says the best
        subset falls monotonically past five — so every number `marginal` prints is the price of a move the
        pin forbids. The champion's own history is swaps: `res392_ws` out and `band_dense` in took 0.925 to
        0.928, and four different members occupying ONE seat is the experiment that produced the ruler
        caveat that pin now carries. The tool that decides which member to ship could not express any of it.

        The BEST seat, not every seat, because the decision is one line: whether this candidate belongs in
        the package at all, and at whose expense. Reporting four rows per candidate turns a roster of four
        against a dozen candidates into forty-eight numbers to eyeball, and the maximum is what a
        submission would take.

        A negative best-seat number is the honest answer that the roster should not be disturbed, and it
        is read CONSERVATIVELY — the opposite of what this docstring used to say (knee-c0u9). The old
        wording put a small negative "inside the region where this ruler is known wrong" on the 2.4x
        diversity under-pricing of knee-7d1, which reads as a licence to ship one. Two entries were fired
        on that theory and both were refuted. That 2.4x came off the DECODE axis, by collapsing three
        members into one pixel group, and it does not carry to a member swap. Three paired entries measure
        the opposite: the holdout RANKS correctly (0.8691 / 0.8678 / 0.8668 offline against 0.930 / 0.926
        / 0.925 on the board) and errs only in MAGNITUDE, understating the cost — -0.0013 offline became
        -0.004 and -0.0023 became -0.005. So `BOARD_COST_MULTIPLE` is what a negative is reported at, and
        a delta inside the noise floor is an argument for NOT spending the slot rather than a free look.

        `BOARD_RESOLVABLE` IS WHERE A POSITIVE STOPS MEANING ANYTHING, and the report says so rather than
        leaving the reader to remember it. It is 0.0040 as of 2026-09-02, raised from 0.0015 because the
        old value was inside the band where this ruler is measured to invert: across the 34 board rows
        carrying an honest offline number — 561 ordered pairs — 35 invert, and the largest offline
        separation at which one still does is 0.0037. So the trailing count is deliberately not a filter:
        the ordering is this tool's product and stays whole, and the line only refuses to let a move the
        record has been seen to order BACKWARDS read as a submission.
        """
        truth = candidates.truth
        base = self.scores.mean(RankAverage().combine(roster.predictions), truth)
        if not candidates.names:
            # EVERY candidate was skipped for having no holdout record, which is a STATE and not an error:
            # `final_ws_e12` is fitted at --fold -1 and can never carry one, so the arm that produces the
            # deliverable ends by asking this question of an empty pool. It used to answer by raising
            # `max() iterable argument is empty` from the line below, AFTER a twelve-epoch fit had
            # written its checkpoint — a crash in a reporting step reads exactly like a training failure
            # in the log, and this one cost a first reading of the run that produced it.
            return ["no candidate carries a holdout record, so no seat can be priced"]
        priced = []
        for candidate, name in zip(candidates.predictions, candidates.names, strict=True):
            seats = [
                (
                    self.scores.mean(
                        RankAverage().combine([*roster.predictions[:seat], candidate, *roster.predictions[seat + 1 :]]),
                        truth,
                    ),
                    roster.names[seat],
                )
                for seat in range(len(roster))
            ]
            score, displaced = max(seats, key=lambda pair: pair[0])
            priced.append((score, name, displaced))
        lines = [f"holding {len(roster)} members, the best seat each candidate could take:"]
        resolvable = sum(1 for score, _, _ in priced if score - base >= BOARD_RESOLVABLE)
        best = max(score for score, _, _ in priced) - base
        return [
            *lines,
            *(
                f"  {name} for {displaced}: {score:.4f}  ({score - base:+.4f})"
                for score, name, displaced in sorted(priced, key=lambda priced: -priced[0])
            ),
            f"{resolvable} of {len(priced)} clear +{BOARD_RESOLVABLE:.4f}, under which this ruler has "
            f"never been shown to pick a board tick",
            self.board_cost(best),
        ]

    @staticmethod
    def board_cost(best: float) -> str:
        """What the board would charge for shipping the best candidate, when the best one is negative.

        SAID ON THE REPORT rather than left in a docstring, because the reading it corrects is the one a
        reader reaches for unprompted: a best seat of -0.0010 sits inside the 0.0062 noise floor and
        looks free. It is not — three paired entries put the board cost at about `BOARD_COST_MULTIPLE`
        times the offline number (knee-c0u9), which turns that -0.0010 into roughly -0.003, still short of
        the +0.0040 the same ruler needs before it stops ordering the board backwards.
        """
        if best >= BOARD_RESOLVABLE:
            return f"best clears +{BOARD_RESOLVABLE:.4f}; above that this ruler orders correctly but saturates"
        if best >= 0:
            return f"best is +{best:.4f}, inside +{BOARD_RESOLVABLE:.4f}: this ruler cannot tell it from the incumbent"
        return (
            f"best is {best:+.4f} and NEGATIVE, which the board charges at about {BOARD_COST_MULTIPLE}x, "
            f"so roughly {best * BOARD_COST_MULTIPLE:+.4f} — not a free look (knee-c0u9)"
        )

    def marginal(self, roster: Pool, candidates: Pool) -> list[str]:
        """What each candidate would add to the roster that already ships, best first.

        THE NUMBER AN ARM IS ACTUALLY DECIDED ON, and the one nothing here computed. Every arm is read
        against a MATCHED control — the same recipe with one field changed — which is the right ruler
        for `did this knob do anything` and the wrong one for `does this earn a seat`. The two came
        apart once the roster stopped being the control's family: a member can beat its control by a
        clear margin and still be the pick that lowers the ensemble.

        `res392_ws` is the case that forces this. It scores 0.8523 alone, it was a member of the four
        that scored 0.925 on the board, and over 22 same-split members the greedy takes it NINTH — the
        first pick that does not raise the mix. Solo score ranked it into a submission; marginal
        contribution never would have.

        Distinct from `greedy`, which builds a roster from nothing and answers which members to
        package. This holds the roster FIXED, because the roster is a shipped artefact with a board
        number on it and the question an arm asks is whether to disturb it.

        It prices a FIFTH member, which the pin's four-member principle says is negative return — so read
        it as the diversity signal (does this member bring anything the four do not) and read `swap` for
        the decision.
        """
        truth = candidates.truth
        combined = RankAverage().combine(roster.predictions)
        base = self.scores.mean(combined, truth)
        added = []
        for candidate, name in zip(candidates.predictions, candidates.names, strict=True):
            widened = RankAverage().combine([*roster.predictions, candidate])
            band = self.scores.paired_mean_interval(combined, widened, truth)
            added.append((self.scores.mean(widened, truth), name, band))
        lines = [f"roster of {len(roster)}: {base:.4f}"]
        lines += [
            f"  +{name}: {score:.4f}  ({score - base:+.4f})  [{low:+.4f}, {high:+.4f}]{'*' if low > 0.0 else ''}"
            for score, name, (low, high) in sorted(added, key=lambda priced: -priced[0])
        ]
        return [*lines, *self.selection_caveat(sum(1 for _, _, (low, _) in added if low > 0.0), len(added))]

    @staticmethod
    def selection_caveat(clearing: int, priced: int) -> list[str]:
        """Said on the report, because the band is exactly the number a reader will over-trust.

        A paired band is honest for ONE PAIR NAMED IN ADVANCE. The top row of this table is not that —
        it is the argmax of however many candidates were priced, and the largest of many draws clears
        zero far more often than any single one of them would. That is `LabelCombine`'s trap in a new
        place, and the whole reason `MemberCurve.honest` reports SPLIT-HALF.

        A star is therefore evidence about the pair and not a licence to ship it. When nothing clears at
        all the table is unambiguous and says so, which is the common case and the useful one.
        """
        if not clearing:
            return [f"no candidate's band excludes zero over {priced} priced — this pool adds nothing readable"]
        return [
            f"{clearing} of {priced} bands exclude zero; the starred TOP row is an argmax over {priced} "
            f"draws, so read it as a candidate for the split-half check, never as a decision"
        ]

    def best(self, pool: Pool) -> list[str]:
        """The winning subset at each size, named — the line a submission is built from."""
        lines = []
        for size in range(1, len(pool) + 1):
            score, subset = max(self.subsets(pool, size), key=lambda pair: pair[0])
            lines.append(f"best {size}: {score:.4f}  {pool.named(subset)}")
        return lines

    def routing(self, pool: Pool) -> list[str]:
        """Per-label member selection at every k, split-half beside in-sample — routing, priced honestly.

        A method rather than the inline call `logged` used to make, because `logged` ALSO enumerates every
        subset and that is 2**n: past a dozen members the only reading of this pool that can run at all is
        this one, and it was reachable only through the reading that cannot. The routing question gets
        SHARPER with pool size — one specialist among a hundred members is exactly what a mean-based seat
        scan hides — so the reading and the enumeration had to come apart.

        Its last row IS the flat rank-average under the same estimator, which is why no separate control
        run is needed and why the two columns may be subtracted.
        """
        return LabelCombine(self.scores).curve(pool.predictions, pool.truth, self.combine_seed)

    def labels(self, pool: Pool) -> list[str]:
        """The MIX's own per-label AUCs, worst first, each priced in units of the final score.

        THE DENOMINATOR NOBODY ASKED THE ENSEMBLE. Every reading in this module collapses to a mean,
        and the competition metric is the unweighted mean of twelve INDEPENDENT AUCs — so a mean that
        will not move can still hide a label sitting near chance while eleven others are saturated.
        Each label carries exactly 1/12 of the score, which makes the arithmetic a one-liner and makes
        it decisive: a single finding moved 0.70 -> 0.78 is +0.0067 on the board, larger than every
        ensemble rearrangement this project has measured put together.

        The `to 0.90` column is the score this mix would post if that ONE label were lifted to 0.90 and
        nothing else changed. It is a headroom, not a forecast: it says where the remaining score
        physically lives, so that a label-side arm is aimed at the finding that can pay for it rather
        than at whichever one is easiest to move.

        `nan` is a label the shared holdout never saw both classes of — reported rather than dropped,
        because an undefined AUC is a hole in the ruler at exactly the rare findings that carry the
        most unrealised score.
        """
        combined = pool.combined(list(range(len(pool))))
        per_label = self.scores.per_label(combined, pool.truth)
        mean = float(np.nanmean(list(per_label.values())))
        weight = 1.0 / len(per_label)
        lines = [f"mix of {len(pool)}: {mean:.4f} over {len(per_label)} labels, each worth {weight:.4f}"]
        # THE DENOMINATOR, ON THE SAME LINE AS THE NUMBER IT QUALIFIES. A low AUC on forty positives and
        # a low AUC on four hundred are different findings — the first can be sampling, the second
        # cannot — and this reading exists to aim an arm, so it must not let the two read alike.
        positives = (pool.truth > MIDPOINT).sum(axis=0)
        counts = dict(zip(per_label, (int(count) for count in positives), strict=True))
        ordered = sorted(per_label.items(), key=lambda pair: (not np.isnan(pair[1]), pair[1]))
        for label, auc in ordered:
            lifted = mean + weight * (LABEL_HEADROOM - auc)
            lines.append(
                f"  {label:18s} {auc:.4f}  n={counts[label]:4d}   "
                f"to {LABEL_HEADROOM:.2f}: {lifted:.4f}  ({lifted - mean:+.4f})"
            )
        return lines

    def paired(self, pool: Pool) -> list[str]:
        """Each member's per-label delta against the FIRST, with the interval that says if it is real.

        Against the first because a paired interval is a two-arm statement and the caller names its
        control by putting it first — the same convention `run_comparison` uses.

        A star means the interval excludes zero. Without one, `run_comparison` prints the UNPAIRED
        bound, which ignores that both arms are scored on the same studies and runs about eleven times
        wider — wide enough that the calibrated-targets arm returned twelve deltas of which none
        cleared, and its whole falsifiable claim was per-label.
        """
        lines = []
        truth = pool.truth
        for index in range(1, len(pool)):
            control = self.scores.per_label(pool.predictions[0], truth)
            observed = self.scores.per_label(pool.predictions[index], truth)
            interval = self.scores.paired_interval(pool.predictions[0], pool.predictions[index], truth)
            lines.append(f"paired deltas, {pool.names[index]} against {pool.names[0]}:")
            for label, (low, high) in interval.items():
                clears = "*" if low > 0.0 or high < 0.0 else ""
                delta = observed[label] - control[label]
                lines.append(f"  {label:18s} {delta:+.4f}  [{low:+.4f}, {high:+.4f}]{clears}")
        return lines

    def oracle(self, pool: Pool) -> dict[str, float]:
        """The CEILING of combining these members, per label — against the best subset actually reachable.

        A ceiling by construction, and it must never be quoted as a score: it picks, for each of the
        twelve findings, whichever member happens to score that finding best ON THIS HOLDOUT. Nothing
        can be selected that way at submission time, because the choice uses the answers.

        WHAT IT IS FOR. It prices a whole class of work before any of it is done. Our current members
        are +0.02 better on this ruler than the five that scored 0.888, and the entry built from them
        scored 0.882 (knee-jk3). Either diversity dominates quality here, or holdout-to-leaderboard
        transfer is not what we assume. If this ceiling sits near the plain rank-average, then per-label
        weighting is worthless and so is another member of the same kind; if it sits far above, the gain
        is already sitting in the members we have and costs no training to collect.

        Reported beside the best PLAIN average so the two are read together — a ceiling alone tempts
        exactly the misreading it exists to prevent.
        """
        truth = pool.truth
        per_label = [self.scores.per_label(member, truth) for member in pool.predictions]
        labels = list(per_label[0])
        best = {label: max(scored[label] for scored in per_label) for label in labels}
        return {
            "oracle": float(np.nanmean(list(best.values()))),
            "average": self.scores.mean(pool.combined(list(range(len(pool)))), truth),
            "best_single": max(self.scores.mean(member, truth) for member in pool.predictions),
        }

    def votes(self, pool: Pool) -> list[str]:
        """Every voting rule on the SAME members — the ensemble axis that has never been varied.

        Count, which members, per-label selection and decode diversity have all been measured. The vote
        itself has not: every number in the ledger came out of a flat per-label rank-average, chosen
        because the metric reads rank order — which argues for RANKING and says nothing about weighting
        six members equally when 0.031 separates the best from the worst.

        The fitted rule is reported SPLIT-HALF, and that is the only column worth reading for it: its
        weights come off the same studies it would otherwise be scored on, and per-label selection died
        of exactly that — an in-sample gain that did not survive being chosen elsewhere.
        """
        flat = RankAverage()
        members, truth = pool.predictions, pool.truth
        lines = [f"{flat.named():22s} {self.scores.mean(flat.combine(members), truth):.4f}   (the shipped rule)"]
        lines.append(
            f"{LogitAverage().named():22s} {self.scores.mean(LogitAverage().combine(members), truth):.4f}"
            "   (keeps the margin a rank discards; exposed to a miscalibrated member)"
        )
        lines.append(
            f"{'QualityWeightedRank':22s} {self.quality_weighted(pool):.4f}   (split-half; weights "
            "fitted on the studies it is NOT scored on)"
        )
        return lines

    def quality_weighted(self, pool: Pool) -> float:
        """The weighted rule's HONEST score: weights fitted on one half, read on the other.

        Its own method rather than three lines inside `votes`, because it is the only rule here that fits
        anything and so the only one whose reported number depends on a split — which is a claim to state
        once, in the place that makes it, rather than a nested comprehension inside a report builder.
        """
        members, truth = pool.predictions, pool.truth
        halves = LabelCombine(self.scores).halves(len(truth), self.combine_seed)
        honest = [
            self.scores.mean(
                QualityWeightedRank([self.scores.mean(member[fit], truth[fit]) for member in members]).combine(
                    [member[report] for member in members]
                ),
                truth[report],
            )
            for fit, report in halves
        ]
        return float(np.mean(honest))

    def logged(self, pool: Pool, *, paired: bool) -> list[str]:
        """Every reading of these members' arrays — the half of this CLI that needs no GPU and no corpus.

        Split from `main` because it is the whole analysis: `main` finds checkpoints, builds a holdout
        and fills a cache, and this reads what came back. Together they made one function that could
        only run with a card, a corpus and minutes, which is to say one that nothing could exercise.
        """
        lines = list(self.report(self.sizes(pool)))
        # The subsets BY NAME, because the mean and the range describe the ensemble question while a
        # submission needs the answer: which members to package.
        lines += self.best(pool)
        if paired:
            lines += self.paired(pool)
        # The CEILING beside the reachable numbers, never alone: it selects per label using the answers,
        # so quoting it as a score would be quoting a cheat. What it prices is whether per-label
        # weighting - or another member of the same kind - is worth anything before either is built.
        ceiling = self.oracle(pool)
        lines.append(
            f"oracle {ceiling['oracle']:.4f}  vs plain average {ceiling['average']:.4f}  "
            f"vs best single {ceiling['best_single']:.4f}  "
            f"(headroom {ceiling['oracle'] - ceiling['average']:+.4f})"
        )
        # THE REACHABLE HALF OF THAT CEILING. The oracle says what per-label selection would be worth if
        # the answers were available; this says what is left once they are not. Its last row is the flat
        # rank-average under the same split-half estimator, so the two columns are a matched comparison
        # rather than a number beside a differently-measured number.
        lines += self.routing(pool)
        # THE VOTE ITSELF, last, because every line above it was produced BY the flat rule and this
        # is the one row that asks whether that rule was the right one to produce them with.
        return lines + self.votes(pool)

    def report(self, by_size: dict[int, list[float]]) -> list[str]:
        """One line per ensemble size: its mean, its range, and what the member added."""
        lines = []
        previous = None
        for size, values in sorted(by_size.items()):
            mean = float(np.mean(values))
            gain = "" if previous is None else f"   {mean - previous:+.4f} over {size - 1}"
            lines.append(f"{size} member(s): mean {mean:.4f}   [{min(values):.4f}, {max(values):.4f}]{gain}")
            previous = mean
        return lines


def main() -> None:
    """Score every subset of the given checkpoints on their shared holdout."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="members sharing ONE split")
    command.parser.add_argument("--labels", type=Path, required=True)
    command.parser.add_argument("--fold", type=int, default=0)
    command.parser.add_argument("--folds", type=int, default=5)
    command.parser.add_argument("--split-seed", type=int, default=2026)
    command.parser.add_argument("--slices", type=int, default=12)
    command.parser.add_argument("--overlap", action="store_true", help="every window start, not just disjoint ones")
    command.parser.add_argument(
        "--paired",
        action="store_true",
        help="per-label 95%% interval of each member's delta against the FIRST, by bootstrap over "
        "studies. The unpaired bound run_comparison prints is ~11x wider because it ignores that both "
        "arms are scored on the same studies - wide enough that a whole arm's per-label claim came "
        "back unread. Costs a thousand rescorings of arrays already in memory, and no GPU.",
    )
    command.parser.add_argument(
        "--tta",
        action="store_true",
        help="average over SpatialViews (rotations 0/-6/+6) as well as windows. A whole-stack member "
        "reads one window, so window averaging - worth a measured +0.0156 to the windowed members - is "
        "structurally unavailable to it and rotation is the only test-time average it can have. Costs "
        "three forward passes per member.",
    )
    command.parser.add_argument(
        "--combine-seed",
        type=int,
        default=COMBINE_SEED,
        help="which half of the holdout fits the per-label member choice and which reports it. Fixed by "
        "default so the curve is reproducible; vary it to see how much of a gain is the split.",
    )
    command.parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-read every member through its encoder instead of using cached probabilities. The cache "
        "keys on the checkpoint's size and mtime, the scored studies and the whole reading policy, so a "
        "stale hit should not be possible - this is the escape hatch for when it is.",
    )
    command.parser.add_argument(
        "--device",
        default=VisibleCards().device().type,
        choices=("cuda", "cpu"),
        help="where to read misses. This analysis runs WHILE the card belongs to a trainer, so saying "
        "cpu explicitly stays available — but the default now answers it too, since `VisibleCards` "
        "counts visible cards rather than asking whether CUDA is installed.",
    )
    arguments = command.parsed()

    root = command.rooted(arguments)
    device = torch.device(arguments.device)
    views = SpatialViews() if arguments.tta else None
    predictions = HoldoutPredictions(device, views)
    scores = LabelScores()

    # EACH MEMBER'S OWN READING, never one cache and one size for all of them. This CLI loaded a single
    # 336 px cache, passed IMAGE_SIZE as the read size for every member, and windowed by GROUP whatever
    # the member was - so a whole-stack member was scored on three slices at a time and a member fitted
    # on the 448 px cache could not be scored at all. It is the tool that decides WHICH members to
    # submit, so a wrong number here chooses the wrong ensemble.
    # KEYED ON THE CACHE'S OWN NAME, which is the only thing that promises two members read the same
    # pixels. This grouped by size, depth and scheme — three of the six fields that name a cache — so
    # two members differing in crop, band or tie-break shared one decode, and the member built its own
    # `StudyCache` from the three it happened to remember.
    # ONLY MEMBERS THAT WERE HELD OUT FROM THESE STUDIES. A member records the fold it held out, and a
    # member whose fold is not the one being scored TRAINED on these studies — `--fold -1` trained on all
    # of them. Scoring it here is in-sample and reads high: `base_dense_ff` came back 0.9035 on 2026-08-30
    # against every honest member's 0.83-0.86. The greedy then seats in score order, so a leaked member is
    # seated FIRST and the whole ordering is junk — silently, in a log that looks like any other sweep.
    # `record_mix` has always refused them; this tool averaged them in (knee-vq46).
    readings = MemberReadings(Member.held_out_from(arguments.fold, arguments.checkpoints), root.processed())

    # The split is taken from the FIRST cache and reused, so every member is scored on the same studies
    # in the same order. Any other cache that disagrees about that order is refused: its rows would be
    # silently misaligned against the truth vector, which is a wrong number rather than an error.
    corpus, reference = readings.reference()
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    _, holdout = folds.holdout(folds.assign(str(root.raw() / "train.csv"), reference), arguments.fold)
    truth = Targets.load(arguments.labels, root.raw() / "train.csv", reference).scores[holdout]

    # Keyed on the studies actually scored, so a different fold or split cannot hand back these rows.
    cache = PredictionCache(
        root.processed() / "holdout_predictions",
        [reference[i] for i in holdout],
        predictions.views,
        refresh=arguments.refresh,
    )

    per_member: list[NDArray[np.float64]] = []
    names: list[str] = []
    # The walk is the long part — every member that is not already cached is a full inference pass, and
    # a sweep over the pool runs unattended for hours past the idle timer (knee-wyzj).
    with Wakelock():
        for member, slab, held in readings.slabs(holdout, reference, corpus):
            window = member.window(held.slices, overlap=arguments.overlap)
            scored = predictions.cached(member, slab, cache, window)
            per_member.append(scored)
            names.append(member.file.name)
            logger.info("%s alone: %.4f", member.file.name, scores.mean(scored, truth))

    pool = Pool(names, per_member, truth)
    command.report(MemberCurve(scores, arguments.combine_seed).logged(pool, paired=arguments.paired))


if __name__ == "__main__":
    main()
