"""How the loss reads a compartment PAIR: as two independent columns, or as two sides to be ordered.

THE MEASUREMENT THIS ANSWERS. `kneemri.evaluation.side_blindness` scores each of a pair's columns at
ranking the studies positive on its own side above those positive on the other side only — a contrast
whose null is 0.50 by identity, because a column that does not depend on the side cannot separate
groups that differ only in the side. Read on the shipped seven-member mix over 909 studies, and
reproduced on all eight members separately:

    Lateral Meniscus 0.7646 · Medial Meniscus 0.8728 · Medial OA 0.7583 · Lateral OA 0.5250

Three of four columns resolve their own side. The fourth is at chance, on every member, under every
recipe measured. A neutral third column (PF OA, which belongs to neither side) reads 0.44-0.47 on the
same contrasts, so the groups do not differ in overall disease burden and the three above-chance
readings are side information rather than a sicker-knee artefact.

WHY THAT POINTS AT THE LOSS AND NOT AT THE PIXELS. The meniscus pair resolves under the SAME encoder,
the same pooling, and the same field of view, so the representation demonstrably carries which side a
finding is on. knee-yrg separately refuted four data-side explanations for the deficit — the laterality
flip, the slice band, prevalence, and the in-plane crop. What is left is that nothing in the objective
ever asks the two columns of a pair to differ. Cross-entropy scores each of the twelve independently,
so a head can minimise it on a pair by emitting the pair's marginal twice, and the existing ranking term
does not close it either: it orders study against study WITHIN one label and never one label against
another.

THE TERM. This is that missing comparison, and it is the mirror of `Objective.pairs` along the other
axis. Within a study, for each annotated compartment pair, the side the reports grade higher is asked to
outrank the side they grade lower, in proportion to how far apart the two readings are and how confident
both were. A study the reports grade equally on both sides contributes nothing and needs no mask; so
does a study where either reading is a guess.

IT IS LABEL STRUCTURE, NOT A PATCH FOR OUR WEAKNESS. The term is defined for every compartment pair in
both directions and knows nothing about which of them we happen to be bad at — it would be written the
same way if the lateral columns were our strongest. What it encodes is that two columns naming the same
lesion on opposite sides carry a within-study ordering that twelve independent binary problems throw
away.

WHAT IT CANNOT DO. It cannot put side information into a representation that has none. If the OA
deficit turns out to be that the encoder resolves meniscus laterality but not OA laterality, this term
has nothing to sharpen and the arm will read flat — which is a real outcome and the reason the arm is
read on the contrast, whose null is exact, rather than on a mean that a 0.0062 floor would swallow.
"""

from typing import override

from jaxtyping import Float
from torch import Tensor, nn

from core.findings import COMPARTMENT_PAIRS, TARGETS
from kneemri.training.grades import Predicted

SIDE_TERM = "side_weight"
"""The key a run's settings answer in, so an arm can check the term actually reached the objective.

Same reason `RANK_TERM` exists: a `--side-weight` that never landed trains the plain recipe under this
arm's name and reports a flat contrast that reads as a refutation of the mechanism.
"""

SIDE_FLOOR = 1e-6
"""Guards a batch in which no study separates the two sides of any pair, so the denominator is zero."""


class SideRanking:
    """The null policy: a compartment pair is two independent columns, which is how we have always fit.

    A real position rather than a placeholder — every member shipped to date trains this way, and the
    side-blindness reading above is a reading OF this policy.
    """

    def term(self, batch: Predicted) -> Float[Tensor, ""]:
        """Nothing to add. Returned as a tensor so the caller adds unconditionally rather than branching."""
        return batch.logits.sum() * 0.0

    def settings(self) -> dict[str, object]:
        return {SIDE_TERM: 0.0}


class PairedSides(SideRanking):
    """Within a study, the worse side of a compartment pair is asked to outrank the better one."""

    def __init__(self, weight: float, pairs: tuple[tuple[str, str], ...] = COMPARTMENT_PAIRS) -> None:
        self.weight = weight
        columns = {target: index for index, target in enumerate(TARGETS)}
        self.columns = [(columns[medial], columns[lateral]) for medial, lateral in pairs]

    @override
    def term(self, batch: Predicted) -> Float[Tensor, ""]:
        """The misordering mass over every pair and both directions, out of the pair mass it is from.

        SUMS POOLED BEFORE DIVIDING, for the reason `Objective.pairs` gives: averaging each pair's own
        ratio would give a pair that separates in three studies the same say as one that separates in
        three hundred. Here that matters more than it does there, because the two pairs have genuinely
        different discordant rates.
        """
        misordered = batch.logits.sum() * 0.0
        mass = batch.logits.sum() * 0.0
        for medial, lateral in self.columns:
            for above, below in ((medial, lateral), (lateral, medial)):
                gap = batch.logits[:, above] - batch.logits[:, below]
                separation = (batch.scores[:, above] - batch.scores[:, below]).clamp(min=0.0)
                pair_weight = batch.weights[:, above] * batch.weights[:, below] * separation
                misordered = misordered + (pair_weight * nn.functional.softplus(-gap)).sum()
                mass = mass + pair_weight.sum()
        return self.weight * misordered / mass.clamp(min=SIDE_FLOOR)

    @override
    def settings(self) -> dict[str, object]:
        return {SIDE_TERM: self.weight}
