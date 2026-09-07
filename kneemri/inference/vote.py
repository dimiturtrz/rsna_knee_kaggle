"""HOW members vote, as a family rather than as one hardcoded rule.

Ensemble design has several axes and only some have been measured. Count is measured (1->2 buys +0.0091,
5->6 buys +0.0008). WHICH members is measured (joint selection over marginal, +0.0024). Per-label
selection is measured and DEAD (its oracle sits 0.0008 BELOW the plain average). Diversity of the decode
path is measured and is the one that transfers (0.0059 offline, 0.014 on the board).

The vote itself had never been varied. Every number this project produced came out of one rule — a flat
per-label rank-average — chosen because the metric reads rank order, which is an argument for RANKING and
not an argument for weighting six members equally when 0.031 separates the best from the worst.

MEASURED, 2026-08-17, and the axis is FLAT. On all six all-ours members: RankAverage 0.8598,
LogitAverage 0.8603, QualityWeightedRank 0.8600 split-half. On the pair that scored 0.919: 0.8581 /
0.8583 / 0.8580.

THAT FIRST RETIREMENT WAS ARGUED WRONGLY, and the correction matters more than the verdict it happened
to preserve. It read those spreads as "an order below the 0.0053 board floor" — comparing an OFFLINE
delta directly against a BOARD threshold, which is the units error knee-7d1 later corrected and
comp/champion.json now forbids by name. At the measured 4.67 board points per holdout point, +0.0005
offline is +0.0023 board, and a lever worth two ticks at zero card cost is not one to dismiss.

RE-PRICED HONESTLY, 2026-08-23, on the champion seven — the pool this project ships, and the most
DISAGREEING one it owns, being the only one holding both the DINOv2-base member and the different-pixels
member. RankAverage 0.8694, LogitAverage 0.8696, QualityWeightedRank 0.8695. The pre-registered gate was
+0.0005 or no submission slot, and +0.0002 fails it.

The mechanism is refuted rather than merely unconfirmed, which is why this axis is now closed instead of
re-openable. The whole case for re-testing was that the effect GROWS with member disagreement; on the
most-disagreeing pool it came out SMALLER (+0.0002) than on the six that share a recipe lineage
(+0.0005). It shrinks. The gains are in the MEMBERS, not in how they are counted — the same conclusion
as 2026-08-17, now resting on an argument that holds. See
interpretations/kneemri/2026-08-23_the_ensemble_axis_is_exhausted.md.

The family stays because the question recurs and answering it again now costs a CLI flag rather than a
re-implementation — and because keeping the rules side by side is what makes the next candidate rule a
two-line addition instead of a fork of the submission path.

`RankAverage` is not a copy of the shipped rule, it IS the shipped rule: `Ensemble.predict_groups` calls
it directly and so does every analysis that reports an ensemble number, so exploring this family cannot
leave the submission path behind and no measured difference between two rules can turn out to be a
difference in how they ranked. There was an `Ensemble.combine` alias in front of it, which is gone: five
analysis modules imported the ensemble — a package, a device and a dict of loaded models — to reach one
pass-through to this class, which is most of what made that module a god-module.
"""

from abc import ABC, abstractmethod
from typing import override

import numpy as np
from numpy.typing import NDArray

CHANCE = 0.5
"""A member scoring this contributes nothing, so it is what a quality weight is measured ABOVE — weighting
by the AUC itself would give a coin-flip member 0.85 of the vote a real member gets."""

PROBABILITY_FLOOR = 1e-6
"""Clipped before a logit, because a member that ever emits exactly 0 or 1 otherwise votes infinity and
carries the ensemble alone. bf16 training and fp16 serving both make a saturated sigmoid reachable."""


SHARPNESS = (2.0, 4.0, 8.0, 16.0)
"""The sharpnesses `vote_ceiling` reads. Legislated, spanning nearly-flat to nearly-binary: the flat rule
is k=0 by construction, and 16 already squashes 90% of the rank range into the two tails, so a grid past
it measures the same two-level vote twice."""

RANK_MIDPOINT = 0.5
"""Where `Vote.ranks` puts the median study, so a squash centred here bends both tails alike."""


class Vote(ABC):
    """One rule for turning several members' per-study, per-label scores into one table.

    A strategy rather than a flag on the ensemble: each rule is a different ARGUMENT about what a member
    contributes, and the arguments share no branch worth writing. The interface takes the members' arrays
    and nothing else — a rule that needs the holdout at combine time is a rule that cannot run at
    submission time, which is the defect that killed per-label selection.
    """

    @abstractmethod
    def combine(self, per_member: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """The ensemble's table, same shape as any one member's."""

    def named(self) -> str:
        """What to print this rule as in a comparison, so a reported number carries the rule that made it."""
        return type(self).__name__

    @staticmethod
    def ranks(member: NDArray[np.float64]) -> NDArray[np.float64]:
        """One member's scores as per-label ranks in [0, 1] — the part of a vote that is per member.

        On the base class because every rank-space rule must rank IDENTICALLY. A weighted average with
        its own ranking would report a difference against the flat one that is partly a difference in
        ranking, and that is the shape of error this project keeps paying for.

        Per label because the metric ranks per label; one ordering across twelve unrelated findings mixes
        them into a single sequence and destroys exactly the information being scored.
        """
        ranked = np.empty_like(member)
        studies = member.shape[0]
        for label in range(member.shape[1]):
            ranked[:, label] = Vote.midranks(member[:, label])
        return ranked / max(studies - 1, 1)

    @staticmethod
    def midranks(column: NDArray[np.float64]) -> NDArray[np.float64]:
        """Tied studies share the AVERAGE of the positions they span, rather than being ordered by index.

        `LabelScores.area_under_curve` was fixed to midranks for this exact defect, and this is the same
        fix on the serving side: under ordinal ranks a member emitting ONE value everywhere — which knows
        nothing — votes a full sweep from 0 to 1 ordered by corpus position, as loudly as a member that
        can actually separate the studies.

        Reachable at submission time rather than hypothetical: serving is fp16, whose eleven mantissa bits
        collapse a saturated sigmoid onto a handful of distinct values. Measured on the greedy-optimal
        seven the training-side tie mass is 7.9e-05 and the blend mean does not move, because float64
        training arrays essentially never tie — so no recorded number changes and only serving does.
        """
        order = np.argsort(column, kind="mergesort")
        _, inverse, counts = np.unique(column[order], return_inverse=True, return_counts=True)
        opens = np.cumsum(counts) - counts
        centres = opens + (counts - 1) / 2.0
        ranked = np.empty_like(column)
        ranked[order] = centres[inverse]
        return ranked


class RankAverage(Vote):
    """The shipped rule: rank per label, average the ranks equally.

    Equally because nothing has yet shown that it should not be — which is what the other rules in this
    module exist to test, and what the flat rule has to survive before any of them replaces it.
    """

    @override
    def combine(self, per_member: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Ranks averaged across members — and a lone member returned untouched.

        Untouched rather than ranked, because ranking one member changes the VALUES a submission carries
        while changing no order, and a rule whose one-member case rewrites the numbers for no gain is a
        rule that will one day be blamed for a difference it did not make.
        """
        if len(per_member) == 1:
            return per_member[0]
        return np.stack([self.ranks(member) for member in per_member]).mean(0)

    @staticmethod
    def agreement(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
        """Two members' orderings correlated per label and averaged over labels.

        Per label and not over the flattened array, for the reason `ranks` is per label: one correlation
        across twelve unrelated findings is dominated by whichever label has the widest prevalence
        spread, and the metric that decides the board scores each label alone.

        IT LIVES HERE, BESIDE THE RANKING IT IS COMPUTED IN, rather than on `MemberDistance` where it was
        written. Two modules need it and only one of them may import the other: `member_distance` imports
        `Pool` from `member_curve`, so a seat rule in `member_curve` that selects by disagreement could
        not reach a function on `MemberDistance` without closing a cycle. Agreement is a property of the
        rank space this rule defines, not of the report that first printed it.
        """
        left_ranks, right_ranks = RankAverage.ranks(left), RankAverage.ranks(right)
        return float(
            np.mean(
                [
                    float(np.corrcoef(left_ranks[:, label], right_ranks[:, label])[0, 1])
                    for label in range(left_ranks.shape[1])
                ]
            )
        )


class SharpenedRank(Vote):
    """Rank-average after a MONOTONE squash of each member's ranks — the same order, a different vote.

    Every rank rule in this module is LINEAR in the ranks: the flat one averages them, the weighted ones
    average them with coefficients. So a member's say is uniform across its own ordering, and moving a
    study from the 10th percentile to the 11th counts exactly as much as moving it from the 49th to the
    50th. That is an assumption nobody here has ever tested, and `column_lift` (2026-09-06) tripped over
    the evidence against it: a synthetic column at AUC 1.0000 was worth +0.0041 when it carried an
    ordinary within-class ordering and +0.0074 when it was BINARY and tied every study inside a class.
    Identical AUC, twice the seat. The rank-average was reading the shape of the ordering, not only the
    ordering.

    This is that shape as a knob. `sharpness` k passes the ranks through a logistic centred on the
    median, so a large k pushes every member toward a two-level vote (the tied binary column that was
    worth double) and a small k leaves the flat rule untouched. Untouched exactly: as k goes to zero the
    logistic becomes affine in the rank, an affine map of every member's ranks averages to an affine map
    of the mean, and AUC is invariant under it. So k is a deviation FROM the shipped rule with the
    shipped rule as its own limit — there is no separate null to fit or trust.

    IT FITS NOTHING, which is what separates it from the weighting family. `QualityWeightedRank` and
    `LabelQualityRank` buy their gain with parameters read off a holdout and have to be split-half tested
    for that reason. k is one legislated number applied identically to every member and every label; its
    honest and fit-on-self readings are the same number by construction.
    """

    def __init__(self, sharpness: float) -> None:
        self.sharpness = sharpness

    @override
    def named(self) -> str:
        """The rule AND its strength, because two sharpnesses are two rules and a bare name hides which ran."""
        return f"SharpenedRank({self.sharpness:g})"

    def squashed(self, ranks: NDArray[np.float64]) -> NDArray[np.float64]:
        """Ranks in [0, 1] bent toward a two-level vote, centred so the median is the pivot."""
        return 1 / (1 + np.exp(-self.sharpness * (ranks - RANK_MIDPOINT)))

    @override
    def combine(self, per_member: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Squashed ranks averaged equally — and a lone member returned untouched, as the flat rule does.

        Untouched for the flat rule's reason and one of its own: with a single member the squash is a
        monotone map of that member's own scores, so it changes every value a submission carries and no
        order at all.
        """
        if len(per_member) == 1:
            return per_member[0]
        return np.stack([self.squashed(self.ranks(member)) for member in per_member]).mean(0)


class QualityWeightedRank(Vote):
    """Rank-average weighted by how far each member beats chance on the holdout.

    The hypothesis this exists to refute: with 0.031 between the best and worst member, a flat vote hands
    the weakest one as much say as the strongest, and a weight proportional to what a member actually
    knows should be worth something.

    The weights are FITTED ON A HOLDOUT, so this rule can overfit one — six numbers against a few hundred
    studies. It has to be read on a split half, exactly as per-label selection was, and per-label
    selection is what failed that test.
    """

    def __init__(self, holdout: list[float]) -> None:
        above = [max(score - CHANCE, 0.0) for score in holdout]
        total = sum(above)
        # A pool where nothing beats chance has no basis for preferring any member, so it votes flat
        # rather than dividing by zero or electing whichever member happens to round highest.
        self.weights = [value / total for value in above] if total > 0 else [1 / len(holdout)] * len(holdout)

    @override
    def combine(self, per_member: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """The same per-label ranks the flat rule builds, averaged with the fitted weights."""
        stacked = np.stack([self.ranks(member) for member in per_member])
        return np.tensordot(np.asarray(self.weights), stacked, axes=(0, 0))


class LogitAverage(Vote):
    """Average in logit space instead of rank space.

    The opposite argument to ranking: a rank throws away HOW confident a member was, so two members that
    agree on the order and disagree on the margin vote identically. Averaging logits keeps the margin, and
    a member that is confidently right pulls harder than one that is barely right.

    It is also the rule most exposed to a miscalibrated member, which is why it is measured rather than
    assumed — nothing here has ever required two members' probabilities to mean the same thing.
    """

    @override
    def combine(self, per_member: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Mean of the members' logits, returned as a probability so every rule has one output space."""
        clipped = np.clip(np.stack(per_member), PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
        logits = np.log(clipped / (1 - clipped))
        return 1 / (1 + np.exp(-logits.mean(0)))


class MedianLogit(Vote):
    """The MEDIAN of the members' logits rather than the mean — the same space, a robust centre.

    Every rule above averages, and an average is the estimator that is optimal when the members' errors
    are symmetric and none of them is an outlier. Neither assumption is free here. A roster that mixes
    decode paths, pixel sizes and fits contains members that are confidently wrong on DIFFERENT cells,
    and one such member moves a six-member mean by a sixth of its own error. The median moves by nothing
    until it flips the ordering of the middle two.

    WHY IT IS WORTH MEASURING RATHER THAN ASSUMING. The gain is exactly the size of the outlier problem
    and no larger: on a roster whose members agree, median and mean are the same number, and the rule
    costs a small loss of information (it discards the extremes even when they are right). So this is a
    rule to reach for when seats become HETEROGENEOUS — which is the direction the roster has been moving
    since decode-path diversity became the only axis with super-linear board transfer — and one to leave
    alone while seats are twins.

    It also breaks a symmetry the other rules share: median in logit space and median in probability
    space pick the SAME member, because the logit is monotone. So unlike the mean, this rule is
    invariant to the space it is computed in, and a miscalibrated member cannot drag it by inflating a
    margin. That is the property `LogitAverage`'s own docstring names as its exposure.

    MEASURED, 2026-08-27, and it LOSES — not by noise, by structure. On the offline shadow of the
    shipped roster (the five holdout twins of the champion six plus the two `pixgroup` adds, 909 shared
    studies): RankAverage 0.8697, LogitAverage 0.8702, MedianLogit 0.8664. −0.0033 against the rule the
    ledger uses and −0.0038 against the rule the board entry serves.

    THE SHAPE OF THE LOSS IS THE FINDING, not the size. Every mean rule's seat curve rises monotonically
    with count; this one OSCILLATES on the parity of the seat count — 2 up, 3 down, 4 up, 5 down, 6 up,
    7 down. An even median averages the middle two members and so keeps some of the mean's information;
    an odd one elects a single member and discards the rest of the pool. So a median vote makes the
    ensemble's score depend on whether the roster happens to be even, which is a property no seating
    argument can control for. The robustness this rule was reached for is real and the pool it was
    reached for is not outlier-shaped enough to pay for it.

    It stays in the family because a measured refutation is worth more standing than deleted — the next
    roster that IS outlier-shaped can re-ask this in one flag, and will have the parity artefact named
    for it in advance.
    """

    @override
    def combine(self, per_member: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Median of the members' logits, returned as a probability so every rule has one output space."""
        clipped = np.clip(np.stack(per_member), PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
        logits = np.log(clipped / (1 - clipped))
        return 1 / (1 + np.exp(-np.median(logits, axis=0)))


class LabelQualityRank(Vote):
    """Rank-average weighted by what each member knows ABOUT EACH LABEL, not about the corpus.

    The ceiling of the weighting family, and the rule that says whether the family is worth anything.
    `QualityWeightedRank` gives a member ONE weight for all twelve findings, which asserts that a member
    good at fractures is good at menisci in the same proportion. The ledger says otherwise: the per-label
    profiles in `weights_gap` disagree about which finding is even the cheapest, and members here are
    fitted at different crops, bands and slice counts — geometry that acts on some findings and not
    others. If weighting pays at all, it pays here, with twelve weights per member instead of one.

    IT IS ALSO THE MOST OVERFITTABLE RULE IN THE MODULE, and that is deliberate. Six members times twelve
    labels is 72 fitted numbers against a few hundred studies per label, so a fit-on-self reading of it
    is guaranteed to look good and means nothing. It exists to be read on a SPLIT HALF by
    `vote_ceiling`, next to its own fit-on-self number, because the gap between those two is the
    measurement — not either one alone. Per-label SELECTION already failed exactly this test, and
    weighting is the softer version of the same idea, so the prior is that it fails too.

    The weights come in already fitted rather than being fitted here, for the reason the `Vote` interface
    names: a rule that needs the holdout at combine time cannot run at submission time.
    """

    def __init__(self, per_label: NDArray[np.float64]) -> None:
        above = np.maximum(per_label - CHANCE, 0.0)
        totals = above.sum(axis=0, keepdims=True)
        # A label no member beats chance on has no basis for preferring any of them, so it votes flat
        # there rather than dividing by zero — per label, because that emptiness is a per-label fact.
        flat = np.full_like(above, 1 / above.shape[0])
        self.weights = np.where(totals > 0, above / np.where(totals > 0, totals, 1.0), flat)

    @override
    def combine(self, per_member: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """The same per-label ranks every rank rule builds, averaged with this label's own weights."""
        stacked = np.stack([self.ranks(member) for member in per_member])
        return (stacked * self.weights[:, np.newaxis, :]).sum(axis=0)


class ColumnWeightedLogit(Vote):
    """`LogitAverage` with a per-member per-column weight instead of the scalar mean.

    The same shape as `LabelQualityRank` — M x 12 weights, fitted elsewhere, handed in through
    `__init__` — in the space `LogitAverage` votes in. That pairing is the point: the rank rule's
    weights are DERIVED from per-label holdout AUC, this one's are FITTED against per-label AUC of the
    weighted logit mean itself, and the two answer the same question with different estimators.

    WHY IT IS A RULE AND NOT A FLAG ON `LogitAverage`. Fifteen call sites construct `LogitAverage()`
    with no arguments; an optional table there would make every one of them a branch, and the module's
    own argument is that keeping the rules side by side is what makes the next candidate a two-line
    addition. A new class leaves every shipped kernel byte-identical without a code path change.

    MEASURED BEFORE IT WAS WRITTEN (D63, researcher, 2026-09-07). Frame 896 L75mid, kernel
    (2/3)*rank(block) + (1/3)*rank(raptor) with the raptor seat fixed, ten candidate members, weights
    softmax-parametrised and non-negative, Nelder-Mead per column, 5 study-folds seed 0:
    OOF macro +0.0028 [+0.0009, +0.0045] against the uniform block, in-sample +0.0060. A 2.1x shrink is
    a fit that survives being taken off its own fold, not one that closes an axis. Four of five folds
    positive. The lift is carried by four columns — LatOA +0.0097, MCL +0.0067, PF OA +0.0062,
    Fracture +0.0061 — and six of twelve columns hand at least .39 of their mass to one member.

    THE HONESTY LINE THAT TRAVELS WITH IT: 120 fitted parameters against 896 studies, fitted on the
    frame it is read on. The OOF number is the one to quote; the in-sample one is what the shrink is
    measured from.
    """

    def __init__(self, per_label: NDArray[np.float64]) -> None:
        if np.any(per_label < 0):
            message = "column weights are a non-negative mixture; a negative weight is a fit, not a vote"
            raise ValueError(message)
        totals = per_label.sum(axis=0, keepdims=True)
        if not np.all(totals > 0):
            message = "every column needs at least one member with mass, or that column has no vote"
            raise ValueError(message)
        # Renormalised rather than trusted: the table arrives from a CSV a fit wrote, and a column that
        # sums to 0.999 would silently shrink every logit in it toward zero — a calibration change
        # wearing a weighting change's clothes.
        self.weights = per_label / totals

    @override
    def combine(self, per_member: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """`LogitAverage`'s logits, summed with this column's own weights instead of averaged flat."""
        clipped = np.clip(np.stack(per_member), PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
        logits = np.log(clipped / (1 - clipped))
        weighted = (logits * self.weights[:, np.newaxis, :]).sum(axis=0)
        return 1 / (1 + np.exp(-weighted))
