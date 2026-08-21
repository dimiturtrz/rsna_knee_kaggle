"""What a readout IS, and how the loss reads the five-level reading that judges it.

The reader does not emit a yes and a no. It emits five levels — 0.08 / 0.28 / 0.68 / 0.82 / 0.94 —
and a policy here decides what a loss is allowed to ask of them: to LAND ON the level's own value, or
merely to fall on the right side of the boundaries between levels. That is a different question from
what the loss then does with the answer, which is why it lives beside the readout rather than inside
`Objective`.
"""

from dataclasses import dataclass
from itertools import pairwise
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor

from kneemri.labels.targets import MEASURED_AGREEMENT

GRADE_LADDER = "grade_ladder"
"""Whether the loss reads the reader's five levels as an ordered ladder rather than as five values."""


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


class Grades:
    """How the loss reads a five-level reading: as one value to hit, which is what we have always done.

    The null half of the pair, and it is a real position rather than a placeholder — every member ever
    shipped here trains this way.
    """

    def spread(self, batch: Predicted) -> Predicted:
        """The batch as it stands: one logit per label, judged against the level's own value."""
        return batch

    def settings(self) -> dict[str, object]:
        return {GRADE_LADDER: False}


class OrdinalGrades(Grades):
    """The same five levels read as four ordered thresholds crossed by ONE latent score.

    WHY THE PARAMETERISATION IS THE SUSPECT, and it is what three previous severity arms all shared.
    The reader emits 0.08 / 0.28 / 0.68 / 0.82 / 0.94 and the plain loss asks a single logit to LAND ON
    each of those values. That spends the logit axis the metric reads — an AUC reads order and nothing
    else — on getting absolute probabilities right for grades the metric never sees. `calib_span` moved
    those values (-0.0020), `severity_weight` moved the weights instead (-0.0048), `severity_flat`
    removed the gradation entirely (-0.0044); all three stayed on one value-fitting logit.

    The cumulative form asks something strictly weaker and exactly as informative: that the score fall
    in an INTERVAL. P(level >= k) = sigmoid(z - t_k), so a severe study only has to outrank threshold 4
    rather than arrive at 0.94. Order is all that is constrained, which is all the metric scores.

    NOTHING DOWNSTREAM CHANGES. The thresholds are subtracted inside the loss, so `z` remains the single
    per-label logit the holdout ranks, the checkpoint stores and the kernel serves. This arm can
    therefore only be a loss result — there is no inference path for it to be confounded by.
    """

    def __init__(self, levels: tuple[tuple[float, float], ...] = MEASURED_AGREEMENT) -> None:
        self.levels = levels

    def boundaries(self) -> list[float]:
        """The score halfway between each pair of adjacent levels — where one grade becomes the next.

        In the reader's own value space, so a target of 0.68 counts as crossing every boundary below it
        no matter what the loss then does with the number.
        """
        assigned = [level for level, _rate in self.levels]
        return [(low + high) / 2 for low, high in pairwise(assigned)]

    def thresholds(self) -> list[float]:
        """Where each boundary sits on the logit axis, DERIVED from what the annotations measured.

        Not tuned and not typed out: threshold k is the logit-space midpoint of the agreement rates of
        the two levels it separates (0.041 / 0.136 / 0.378 / 0.688 / 0.958). Those rates are monotone
        over a span of 0.58, so the thresholds come out ordered by construction — which is the property
        a cumulative link needs and the one a free parameter would have to be constrained into.
        """
        measured = [torch.logit(torch.tensor(rate)).item() for _level, rate in self.levels]
        return [(low + high) / 2 for low, high in pairwise(measured)]

    @override
    def spread(self, batch: Predicted) -> Predicted:
        """One (b, 12) problem restated as four, stacked along the label axis.

        Stacked rather than summed so that `per_label` normalisation still sees each column separately:
        a threshold only a handful of studies cross must not be drowned by the one almost everyone does.
        The weight is the cell's own, repeated — how much a reading is trusted does not depend on which
        boundary is being asked about.
        """
        logits = torch.cat([batch.logits - offset for offset in self.thresholds()], dim=1)
        scores = torch.cat([(batch.scores > boundary).to(batch.scores.dtype) for boundary in self.boundaries()], dim=1)
        weights = batch.weights.repeat(1, len(self.thresholds()))
        return Predicted(logits=logits, scores=scores, weights=weights)

    @override
    def settings(self) -> dict[str, object]:
        return {GRADE_LADDER: True}
