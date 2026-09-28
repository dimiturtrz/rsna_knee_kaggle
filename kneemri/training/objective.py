"""What the training loss actually optimises, as a policy the trainer holds.

The competition scores the unweighted mean of twelve per-label ROC AUCs. An AUC is a **pairwise
ordering** statistic: for one label it is the probability that a randomly chosen positive study is
scored above a randomly chosen negative one. Nothing about the absolute value of a probability is read.

Cross-entropy does not optimise that. It optimises calibration — being right about *how likely* — and
ordering comes out as a by-product. A model can lose real cross-entropy while ranking every pair
correctly, and it spends capacity on the part of the answer the metric throws away.

So an objective here is ONE BASE plus a LIST OF TERMS, each a `Term` strategy summed at its own
strength: `Objective(base, terms)`. The base is confidence-weighted cross-entropy by default, or the
AUC-margin surrogate that REPLACES it (`kneemri.training.terms.Margin`); the terms add to it — the
ranking term, the teacher's ordering, the dissent re-weighting, the contrastive term, the feature
cosine. `Remembered(term, capacity)` widens any `Pooled` term across steps. The terms live in
`kneemri.training.terms`; this module holds the frame they plug into.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from jaxtyping import Float
from torch import Tensor, nn

from kneemri.training.grades import Grades, Predicted
from kneemri.training.label_focus import LabelFocus
from kneemri.training.side_ranking import SideRanking
from kneemri.training.terms import (
    CONTRAST_TERM,
    DISSENT_TERM,
    DISTILL_TERM,
    EARLY_TERM,
    FEATURE_TERM,
    MARGIN_TERM,
    PAIR_POOL,
    RANK_TERM,
    CrossEntropy,
    Term,
)


@dataclass(frozen=True)
class LabelStructure:
    """The policies that read STRUCTURE in the label set rather than treating the twelve as flat.

    They travel together because they answer the same question from different directions. `grades` is
    the ordering WITHIN one label — a report's severity ladder, which a binary target flattens. `sides`
    is the ordering ACROSS two labels of one compartment pair — which side holds the finding, which
    twelve independent binary problems throw away. `focus` is the narrower question underneath both:
    WHETHER a label enters the loss at all, which is what separates a twelve-label panel from the
    specialist models the published numbers come from. All default to the null policy, which is how
    every member shipped to date was fitted, so an objective built without them descends the scalar it
    always did.

    One held object rather than three parameters, so a new policy reaches every objective through the
    one `Shaping` it already takes.
    """

    grades: Grades = field(default_factory=Grades)
    sides: SideRanking = field(default_factory=SideRanking)
    focus: LabelFocus = field(default_factory=LabelFocus)

    def settings(self) -> dict[str, object]:
        """Every policy's account, merged, so a run can check which of them landed."""
        return {**self.grades.settings(), **self.sides.settings(), **self.focus.settings()}

    def focused(self, batch: Predicted) -> Predicted:
        """The batch as the focus policy narrows it, which under the null policy is the batch itself."""
        return self.focus.focused(batch)

    def spread(self, batch: Predicted) -> Predicted:
        """The batch as the grade policy restates it, which under the null policy is the batch itself."""
        return self.grades.spread(batch)

    def side_term(self, batch: Predicted) -> Float[Tensor, ""]:
        """What the side policy adds to the descended scalar, a hard zero under the null policy."""
        return self.sides.term(batch)


PER_LABEL = "per_label_loss"
COUNT_NORMALISED = "normalise_by_count"
"""The two pooling switches every objective reports, beside the term keys `terms` defines."""


@dataclass(frozen=True)
class Shaping:
    """How the loss is normalised, whether it is read per-label, and the label policies it composes.

    The objective UNPACKS it into flat fields rather than holding it, so nothing reads a policy through
    two dots — `self.structure` is used on every line of `__call__`.
    """

    normalise_by_count: bool = False
    per_label: bool = False
    structure: LabelStructure = field(default_factory=LabelStructure)


class Objective:
    """One base plus a list of terms, each summed at its own strength.

    The base is read on the GRADED batch — the grade ladder restates twelve latent scores as threshold
    crossings, and that is a question about calibration. The terms read the twelve latent scores, since
    ordering and representation are separable from grades and only one of them is about grades.
    """

    def __init__(self, base: Term | None = None, terms: Sequence[Term] = (), shaping: Shaping | None = None) -> None:
        settled = shaping or Shaping()
        self.normalise_by_count = settled.normalise_by_count
        self.per_label = settled.per_label
        self.structure = settled.structure
        self.base = base or CrossEntropy(per_label=settled.per_label, normalise_by_count=settled.normalise_by_count)
        self.terms = tuple(terms)

    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """The scalar this step descends.

        The focus mask is applied FIRST: a zeroed weight has to be zero before any term reads it, and
        every term normalises by the weights it was given, so the surviving label descends at its natural
        scale. The side term is added here for every objective alike; its null policy contributes a hard
        zero, so an arm that did not ask for it descends the identical scalar it always did.
        """
        focused = self.structure.focused(batch)
        total = self.base.graded(focused, self.structure.spread) + self.structure.side_term(focused)
        for term in self.terms:
            total = total + term.weight * term(focused)
        return total

    def settings(self) -> dict[str, object]:
        """This objective's own account of the loss switches it carries.

        Every term absent from the list is reported OFF rather than omitted: the question asked before a
        run starts is whether a ranking arm got a ranking term at all, and a key the answer leaves out
        cannot answer it.
        """
        account: dict[str, object] = {
            RANK_TERM: 0.0,
            EARLY_TERM: 0.0,
            DISTILL_TERM: 0.0,
            FEATURE_TERM: 0.0,
            DISSENT_TERM: 0.0,
            MARGIN_TERM: 0.0,
            CONTRAST_TERM: 0.0,
            PAIR_POOL: 0,
            PER_LABEL: self.per_label,
            COUNT_NORMALISED: self.normalise_by_count,
            **self.structure.settings(),
        }
        for part in (self.base, *self.terms):
            account.update(part.settings())
        return account

    def attached(self, device: torch.device) -> list[nn.Parameter]:
        """Every parameter the base and the terms own, moved to the card, for the optimiser to step."""
        return [parameter for part in (self.base, *self.terms) for parameter in part.attached(device)]
