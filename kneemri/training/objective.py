"""What the training loss actually optimises, as a policy the trainer holds.

The competition scores the unweighted mean of twelve per-label ROC AUCs. An AUC is a **pairwise
ordering** statistic: for one label it is the probability that a randomly chosen positive study is
scored above a randomly chosen negative one. Nothing about the absolute value of a probability is read.

Cross-entropy does not optimise that. It optimises calibration — being right about *how likely* — and
ordering comes out as a by-product. A model can lose real cross-entropy while ranking every pair
correctly, and it spends capacity on the part of the answer the metric throws away.

So the two objectives here are a genuine pair rather than a knob:

- `Objective` — confidence-weighted cross-entropy, our measured default.
- `RankedObjective` — the same, plus a term that pays only for pairs ordered the wrong way.

The ranking term is written for OUR supervision, not copied: our targets are weighted confidences in
[0, 1] rather than binary labels, so a pair contributes in proportion to how much the reports actually
separate the two studies and how confident both of those readings were. A pair the reports disagree
about only slightly, or that rests on a guess, pulls proportionally less.
"""

from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor, nn

RANK_WEIGHT = 1.0
"""How hard the ranking term pulls relative to cross-entropy when it is enabled at all.

Both terms are means over their own populations and both sit around 0.7 at initialisation, so equal
weighting is the neutral starting point rather than a tuned one. It is a flag with an explicit default
rather than a swept hyper-parameter: sweeping it before knowing the term helps at all would be
measuring a method that has not yet shown signal.
"""

PAIR_FLOOR = 1e-6
"""Guards a batch in which every study carries the same reading for a label, so no pair separates."""


@dataclass(frozen=True)
class Predicted:
    """One step's readout beside the weighted reading that judges it.

    The three travelled as three arguments through every method here and through the trainer, which is
    what makes a fourth one — an ordinal grade, say — a change to five signatures rather than a field.
    They are also inseparable by construction: a logit judged against another batch's score is not a
    smaller mistake than a missing one.
    """

    logits: Float[Tensor, "b o"]
    scores: Float[Tensor, "b o"]
    weights: Float[Tensor, "b o"]


class Objective:
    """Confidence-weighted cross-entropy, normalised either by total weight or by element count.

    BY WEIGHT (ours): a batch carrying little confidence does not take a full-sized step on almost no
    evidence. BY COUNT (the public baseline's `.mean()`): the step size is a fixed scale that does not
    depend on how confidently the batch happened to be labelled.

    The two are not interchangeable in magnitude. With most weights below one, dividing by their sum
    makes our gradients roughly twice theirs for the same batch — which interacts with the learning-rate
    schedule, so the pair has to be measured together rather than one at a time.
    """

    def __init__(self, *, normalise_by_count: bool = False, per_label: bool = False) -> None:
        self.normalise_by_count = normalise_by_count
        self.per_label = per_label

    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """The scalar this step descends."""
        return self.cross_entropy(batch)

    def cross_entropy(self, batch: Predicted) -> Float[Tensor, ""]:
        """The weighted BCE term, which every objective here is built on top of."""
        elementwise = nn.functional.binary_cross_entropy_with_logits(batch.logits, batch.scores, reduction="none")
        weights = batch.weights
        weighted = elementwise * weights
        if self.per_label:
            # EACH OF THE TWELVE CONTRIBUTES EQUALLY, because that is what the metric does. Pooling the
            # whole batch makes a label's share of the gradient proportional to how often it is
            # mentioned: Effusion is positive in 60% of studies and Fracture in 7%, so the objective
            # spends roughly ten times more on one than the other while the score averages them the
            # same. Normalising per COLUMN and then averaging removes that entirely.
            #
            # It does not touch positives-versus-negatives within a label — a separate question, and a
            # weaker one for a rank metric, since AUC is invariant to the positive rate.
            return (weighted.sum(dim=0) / weights.sum(dim=0).clamp(min=1.0)).mean()
        if self.normalise_by_count:
            return weighted.mean()
        return weighted.sum() / weights.sum().clamp(min=1.0)


class RankedObjective(Objective):
    """Cross-entropy plus a pairwise term that pays only for studies ordered the wrong way.

    Kept as cross-entropy PLUS a term rather than as a ranking loss alone. A pure ranking objective
    fixes nothing about the scale of the logits, which leaves the sigmoid free to drift somewhere flat
    and the gradient with it; and the twelve labels are then coupled only through the encoder, so a rare
    finding with few separating pairs in a batch would contribute almost nothing to any step.
    """

    def __init__(
        self, weight: float = RANK_WEIGHT, *, normalise_by_count: bool = False, per_label: bool = False
    ) -> None:
        super().__init__(normalise_by_count=normalise_by_count, per_label=per_label)
        self.weight = weight

    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """Cross-entropy, plus the ranking term at its configured strength."""
        return self.cross_entropy(batch) + self.weight * self.ranking(batch)

    def ranking(self, batch: Predicted) -> Float[Tensor, ""]:
        """A soft, confidence-weighted count of the pairs this batch ranks the wrong way round.

        Every ordered pair (i, j) within a label is formed at once — the batch is small and the labels
        are twelve, so the whole (b, b, o) comparison is one broadcast rather than a loop over labels.

        A pair counts in proportion to `w_i * w_j * relu(s_i - s_j)`: it is directional (only pairs the
        reports say should go one way are asked to go that way, and the reverse pair carries the mirror
        term), it vanishes when the two readings agree, and it vanishes when either reading is a guess.
        Pairs a study forms with itself have `s_i - s_j = 0` and so drop out without a mask.

        `softplus(-(l_i - l_j))` is the smooth stand-in for "is i above j": zero when i leads by a lot,
        linear in the gap when it trails. That is the surrogate an AUC's step function needs to become
        differentiable, and it is the same one the RankNet family uses.
        """
        logits, scores, weights = batch.logits, batch.scores, batch.weights
        gap = logits.unsqueeze(1) - logits.unsqueeze(0)
        separation = (scores.unsqueeze(1) - scores.unsqueeze(0)).clamp(min=0.0)
        confidence = weights.unsqueeze(1) * weights.unsqueeze(0)
        pair_weight = confidence * separation
        misordered = pair_weight * nn.functional.softplus(-gap)
        return misordered.sum() / pair_weight.sum().clamp(min=PAIR_FLOOR)
