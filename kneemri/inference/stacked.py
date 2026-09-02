"""A LEARNED combiner — the first rule in this project that fits anything to the data.

WHY THIS EXISTS. `kneemri/inference/vote.py` holds five rules and every one of them is FIXED: rank and
average, average logits, take the median, weight by a quality number computed elsewhere. None fits.
`comp/champion.json` says in its own words that the member-quality lane ceilings near 0.943 against a
leader at 0.952 and that the routes left are external data and **stacking** — "whether directional gains
ADD, which nothing has tested and which the whole 0.950 plan assumes". This is the combiner that test
needs; `kneemri/evaluation/stack_fit.py` is the test itself, and it lives there rather than here because
a rule may not import the ruler that judges it.

WHAT A STACKER CAN BUY, AND WHY THE PRIOR IS THAT IT BUYS LITTLE HERE. Krogh-Vedelsby splits an
ensemble's error into the members' average error minus their ambiguity. A fixed average spends the
ambiguity uniformly; a fitted combiner spends it per member and per label, which is strictly more
expressive. But `competence_axis` reads 81.7-82% of what separates 105 members as ONE scalar over 909
studies, and a pool that is one scalar has little ambiguity to redistribute — the extra expressiveness
then has noise to fit instead. That mechanism predicts a small positive, which is what was measured:
+0.0015 / +0.0021 / +0.0009 over the best fixed rule at 6 / 12 / 24 members, all under the 0.0062 floor.

THE LIMIT. Weights can only be fitted where holdout predictions exist, which is fold-0 members. The pool
we SHIP is six `--fold -1` members with no offline predictions at all, so weights reach it only under the
twin assumption `comp/kernels/kneemri-aucvote` states for its per-label AUCs.
"""

import numpy as np
from jaxtyping import Float, Int
from numpy.typing import NDArray
from typing_extensions import override

from kneemri.inference.vote import PROBABILITY_FLOOR, Vote

STACK_PENALTY = 10.0
"""The default ridge penalty on the stacker's member weights. A stated default rather than a tuned one:
`stack_fit` chooses among `PENALTIES` inside its own fitting half, and a value tuned anywhere else would
be a parameter fitted on the studies that judge it."""

PENALTIES = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
"""The grid a caller searches. A grid rather than one legislated value because the right shrinkage depends
on how many members are being combined, which is the axis the measurement varies."""


class StackedLogit(Vote):
    """A per-label ridge over member logits — one weight per (member, label), fitted rather than assumed.

    Per label because the metric is per label, and a member strong on Baker's and weak on MCL is exactly
    the structure one scalar weight per member cannot represent. Ridge rather than logistic because the
    target is an area under a curve, invariant to any monotone map of the score, so a link function buys
    nothing and costs a convergence question.

    Holds its weights and nothing else: a `Vote` needing the holdout at combine time could not run inside
    a kernel, which `Vote`'s own docstring records as the defect that killed per-label selection.
    """

    def __init__(self, weights: Float[NDArray[np.float64], "members findings"]) -> None:
        self.weights = weights

    @staticmethod
    def logits(
        scores: Float[NDArray[np.float64], "studies findings"],
    ) -> Float[NDArray[np.float64], "studies findings"]:
        """Probabilities to logits, clipped, so a saturated member cannot vote infinity."""
        clipped = np.clip(scores, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
        return np.log(clipped / (1.0 - clipped))

    @override
    def combine(self, per_member: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Weighted sum of member logits, per label."""
        stacked = np.stack([StackedLogit.logits(member) for member in per_member])
        return (stacked * self.weights[:, np.newaxis, :]).sum(axis=0)

    @classmethod
    def fitted(
        cls,
        per_member: list[Float[NDArray[np.float64], "studies findings"]],
        truth: Int[NDArray[np.int64], "studies findings"],
        penalty: float = STACK_PENALTY,
    ) -> "StackedLogit":
        """Closed-form ridge weights per label, on the studies handed in and no others.

        BOTH SIDES ARE CENTRED, and the first spelling of this got it wrong in a way worth recording.
        Centring only the target while penalising an uncentred design does not fit an intercept-free
        model — it shrinks the weights toward zero about the wrong origin, and the result LOST 0.0173 on
        held-out studies. That was nearly written down as "stacking is refuted on a correlated pool". It
        was not overfitting: the IN-SAMPLE number was 0.8616, also below the fixed rule's held-out 0.8760,
        and a combiner that loses on the studies it fitted on is misfitted rather than overfitted. Report
        the in-sample number beside the held-out one — it is what catches this.

        No intercept is then needed rather than merely omitted: the output is only ever ranked, so a
        per-label constant is invisible to the score once both sides share an origin.
        """
        stacked = np.stack([StackedLogit.logits(member) for member in per_member])
        members, _, findings = stacked.shape
        weights = np.empty((members, findings), dtype=np.float64)
        for label in range(findings):
            design = stacked[:, :, label].T
            design = design - design.mean(axis=0, keepdims=True)
            centred = truth[:, label].astype(np.float64) - truth[:, label].mean()
            gram = design.T @ design + penalty * np.eye(members)
            weights[:, label] = np.linalg.solve(gram, design.T @ centred)
        return cls(weights)
