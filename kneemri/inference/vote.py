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
            ranked[:, label] = member[:, label].argsort().argsort() / max(studies - 1, 1)
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
