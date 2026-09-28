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

from core.data.labels.targets import MEASURED_AGREEMENT

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
    shaped: bool = False
    """Whether a label policy has already re-weighted these columns, so a second pass leaves them alone.

    `LabelFocus` may be applied at more than one entry (a caller focusing a batch before handing it to
    an `Objective` that focuses again), which a zero/one mask survives because zero and one square to
    themselves. An UNFOCUSED WEIGHT strictly between them does not: applying 0.25 twice descends 0.0625,
    a mechanism landing at the wrong strength with every gate green. The marker makes the contract
    explicit rather than accidental.
    """

    teacher: Float[Tensor, "b o"] | None = None
    """What an ensemble said about these same studies, when a run is distilling — `None` otherwise.

    A FIELD rather than a fourth argument through the chain, for the reason the class docstring gives
    about the ordinal grade. It defaults to `None` so every arm in the ledger constructs the identical
    three-field readout it always did, and so a distillation term that finds nothing here raises rather
    than silently descending cross-entropy under a distilling arm's name.

    It is the TEACHER'S SCORES, in the same [0, 1] space as `scores`, and it is judged against the same
    `logits`: `PairMass.between` reads separation from scores and the gap from logits, so pointing it at
    this field asks the student to reproduce the teacher's ORDER rather than its probabilities — which
    is the only part of a teacher an AUC can read.
    """

    embedding: Float[Tensor, "b o h"] | None = None
    """What each diagnosis READ from the study, before the readout collapsed it to one number.

    `Model.Readout.context`, carried here so a loss can shape the representation rather than only the
    decision. Every term above this one is a function of `logits`, which is a per-finding dot product
    against one learned direction: two studies whose readings differ can share that projection, and
    nothing in a logit says whether the head is separating them for the right reason.

    A FIELD rather than a fifth argument, for the reason `teacher` gives, and `None` by default so every
    member in the ledger constructs the identical readout it always did.
    """

    early: Float[Tensor, "b o 2"] | None = None
    """What the model itself said about these studies on EARLIER visits — `None` unless a run regularises
    toward it. The last axis holds the running averages of `p` and of `1 - p`; see `EarlyLearning`."""

    teacher_embedding: Float[Tensor, "b d"] | None = None
    """What a TEACHER read from these same studies, pooled to one vector per study — `None` otherwise.

    SEPARATE from `teacher`, and not a widening of it, because the two are different objects: that one
    is twelve scores in [0, 1] and this one is a `d`-wide representation whose width is the teacher's
    own and has no reason to equal the student's. A term reading this must project before it compares,
    which is why `FeatureDistillation` owns a linear map and every other term here owns none.

    One vector per STUDY rather than one per finding. The teacher pools its windows once per finding
    and this is the mean over that axis, so the target is what the teacher read about the knee rather
    than the twelve directions its own head happened to split that reading along — the second would
    teach the student the teacher's head, which is the level `swin_distill_fit` measured at -0.0077.
    """


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
