"""What the training loss actually optimises, as a policy the trainer holds.

The competition scores the unweighted mean of twelve per-label ROC AUCs. An AUC is a **pairwise
ordering** statistic: for one label it is the probability that a randomly chosen positive study is
scored above a randomly chosen negative one. Nothing about the absolute value of a probability is read.

Cross-entropy does not optimise that. It optimises calibration — being right about *how likely* — and
ordering comes out as a by-product. A model can lose real cross-entropy while ranking every pair
correctly, and it spends capacity on the part of the answer the metric throws away.

So the objectives here are a chain rather than a set of knobs, each one the last plus one term:

- `Objective` — confidence-weighted cross-entropy, our measured default.
- `RankedObjective` — the same, plus a term that pays only for pairs ordered the wrong way.
- `DistilledObjective` — the same, plus that term judged against an ensemble's ordering.
- `DecorrelatedObjective` — the same term, with its pairs re-weighted toward the ones an ensemble
  orders WRONG. The teacher file of the one above, read for the opposite purpose.

The ranking term is written for OUR supervision, not copied: our targets are weighted confidences in
[0, 1] rather than binary labels, so a pair contributes in proportion to how much the reports actually
separate the two studies and how confident both of those readings were. A pair the reports disagree
about only slightly, or that rests on a guess, pulls proportionally less.
"""

from dataclasses import dataclass, field
from typing import override

from jaxtyping import Float
from torch import Tensor, nn

from kneemri.training.grades import Grades, Predicted
from kneemri.training.label_focus import LabelFocus
from kneemri.training.side_ranking import SideRanking


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

    One held object rather than two parameters: the objective chain is three constructors deep and each
    layer adds its own flag, so threading each policy separately puts `RememberedRanking` over the
    argument count before the next policy is even written.
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


RANK_WEIGHT = 1.0
"""How hard the ranking term pulls relative to cross-entropy when it is enabled at all.

Both terms are means over their own populations and both sit around 0.7 at initialisation, so equal
weighting is the neutral starting point rather than a tuned one. It is a flag with an explicit default
rather than a swept hyper-parameter: sweeping it before knowing the term helps at all would be
measuring a method that has not yet shown signal.
"""

PAIR_FLOOR = 1e-6
"""Guards a batch in which every study carries the same reading for a label, so no pair separates."""

DISTILL_WEIGHT = 1.0
"""How hard the teacher's ordering pulls relative to cross-entropy when distillation is enabled at all.

Equal weighting for the reason `RANK_WEIGHT` gives: both terms are means over their own pair
populations and both sit near `softplus(0)` at initialisation, so parity is the neutral start rather
than a tuned one. Sweeping it before the term has shown any signal would be measuring a method that has
not yet worked.
"""

DISSENT_WEIGHT = 1.0
"""How hard the disagreement term pulls relative to cross-entropy when it is enabled at all.

Parity with cross-entropy for `RANK_WEIGHT`'s reason, and with one more argument specific to this term:
it is a RE-WEIGHTED `ranking`, mass-normalised by the same emphasis that scales it, so it sits on the
same scale as the ranking term whose weight has already been measured neither inert nor dominant here.
"""

RANK_TERM = "rank_weight"
DISTILL_TERM = "distill_weight"
DISSENT_TERM = "dissent_weight"
PAIR_POOL = "pair_memory"
PER_LABEL = "per_label_loss"
COUNT_NORMALISED = "normalise_by_count"
"""The vocabulary an objective answers `settings()` in, so a run can check its loss switches landed.

The loss is chosen by a chain of `if`s over four flags, and every branch of it is a working objective:
a `--rank-weight` that never reached the constructor trains plain cross-entropy under a ranking arm's
name and reports a number that reads as a refutation of the mechanism.
"""

PAIR_MEMORY = 8
"""How many past steps' readouts a `RememberedRanking` keeps, when it is enabled at all.

Set by the two quantities that actually trade here. UPWARD: pairs. A batch of 12 forms 66 unordered
pairs per label and most of them separate by nothing, while eight remembered steps put 96 more studies
on the other side of every comparison — about seventeen times the pair mass, which is the whole point.
DOWNWARD: staleness. A remembered logit was produced by weights that have since taken up to eight
steps, and a pair judged against a model that no longer exists is noise. Eight steps is under 3% of an
epoch's 291 here, and the encoder moves at 8e-6, so the drift over the window is far below the gap
`softplus` is measuring.

It is a flag with an explicit default, not a swept hyper-parameter, for the reason RANK_WEIGHT gives.
"""


class Objective:
    """Confidence-weighted cross-entropy, normalised either by total weight or by element count.

    BY WEIGHT (ours): a batch carrying little confidence does not take a full-sized step on almost no
    evidence. BY COUNT (the public baseline's `.mean()`): the step size is a fixed scale that does not
    depend on how confidently the batch happened to be labelled.

    The two are not interchangeable in magnitude. With most weights below one, dividing by their sum
    makes our gradients roughly twice theirs for the same batch — which interacts with the learning-rate
    schedule, so the pair has to be measured together rather than one at a time.
    """

    def __init__(
        self,
        *,
        normalise_by_count: bool = False,
        per_label: bool = False,
        structure: LabelStructure | None = None,
    ) -> None:
        self.normalise_by_count = normalise_by_count
        self.per_label = per_label
        self.structure = structure or LabelStructure()

    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """The scalar this step descends.

        The side term is added HERE rather than in each subclass, so it composes with all three
        objectives without adding a fourth branch to the chain that builds them: a policy the objective
        HOLDS, exactly as the grade ladder is. The null policy contributes a hard zero, so an arm that
        did not ask for it descends the identical scalar it always did.

        The focus mask is applied at the same place and for the same reason, and it is applied FIRST:
        a zeroed weight has to be zero before any term reads it, and both terms below normalise by the
        weights they were given, so the surviving label descends at its natural scale.
        """
        focused = self.structure.focused(batch)
        return self.cross_entropy(focused) + self.structure.side_term(focused)

    def settings(self) -> dict[str, object]:
        """This objective's own account of the loss switches it carries.

        The plain one reports the ranking term, the distillation term and the pair pool as OFF rather
        than omitting them: the question asked before a run starts is whether a ranking arm got a
        ranking objective at all, and a key the plain answer leaves out cannot answer it.
        """
        return {
            RANK_TERM: 0.0,
            DISTILL_TERM: 0.0,
            DISSENT_TERM: 0.0,
            PAIR_POOL: 0,
            PER_LABEL: self.per_label,
            COUNT_NORMALISED: self.normalise_by_count,
            **self.structure.settings(),
        }

    def cross_entropy(self, batch: Predicted) -> Float[Tensor, ""]:
        """The weighted BCE term, which every objective here is built on top of.

        The grade policy is applied HERE and not in `__call__`, so a ranking term keeps comparing the
        twelve latent scores rather than the forty-eight threshold crossings a ladder restates them as.
        The two questions are separable and only one of them is about grades.
        """
        graded = self.structure.spread(batch)
        elementwise = nn.functional.binary_cross_entropy_with_logits(graded.logits, graded.scores, reduction="none")
        weights = graded.weights
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
        self,
        weight: float = RANK_WEIGHT,
        *,
        normalise_by_count: bool = False,
        per_label: bool = False,
        structure: LabelStructure | None = None,
    ) -> None:
        super().__init__(normalise_by_count=normalise_by_count, per_label=per_label, structure=structure)
        self.weight = weight

    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """Cross-entropy and the side term, plus the ranking term at its configured strength.

        The ranking term is handed the FOCUSED batch. Its pair mass is `weights_i * weights_j`, so a
        zeroed column forms no pairs and contributes to neither the numerator nor the denominator —
        which is what makes one mask on the weights enough to narrow the whole objective. `super()`
        applies the same mask again to its own terms, harmlessly, because the mask is idempotent.
        """
        focused = self.structure.focused(batch)
        return super().__call__(focused) + self.weight * self.ranking(focused)

    @override
    def settings(self) -> dict[str, object]:
        """The base account with the ranking term's real strength, not the flag that asked for it."""
        return {**super().settings(), RANK_TERM: self.weight}

    def ranking(self, batch: Predicted) -> Float[Tensor, ""]:
        """A soft, confidence-weighted count of the pairs this batch ranks the wrong way round.

        Every ordered pair (i, j) within a label is formed at once — the batch is small and the labels
        are twelve, so the whole (b, b, o) comparison is one broadcast rather than a loop over labels.
        """
        misordered, pair_weight = self.pairs(batch, batch)
        return misordered / pair_weight.clamp(min=PAIR_FLOOR)

    def pairs(
        self,
        above: Predicted,
        below: Predicted,
        emphasis: Float[Tensor, "b b o"] | None = None,
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        """The misordering mass asking `above` to outrank `below`, and the pair mass it is out of.

        `emphasis` scales the mass of each pair INDIVIDUALLY, which is the one thing a caller cannot
        express by editing the two `Predicted`s it passes in: `weights` are per study, and "how much
        does this COMPARISON matter" is per pair. It defaults to None, so every objective written before
        it forms the identical mass it always did. It multiplies BOTH returned sums, so a caller
        dividing one by the other gets a re-weighted average of the same surrogate rather than a term on
        a new scale — a uniform emphasis, whatever its size, is exactly no emphasis.

        Returned unreduced as a (numerator, denominator) pair rather than as their ratio, because a
        caller pooling several populations of pairs must add the two sums BEFORE dividing. Averaging
        ratios instead would give a handful of within-batch pairs the same say as a hundred remembered
        ones, which is the opposite of the point.

        A pair counts in proportion to `w_i * w_j * relu(s_i - s_j)`: it is directional (only pairs the
        reports say should go one way are asked to go that way, and the reverse pair carries the mirror
        term), it vanishes when the two readings agree, and it vanishes when either reading is a guess.
        Pairs a study forms with itself have `s_i - s_j = 0` and so drop out without a mask — which is
        also why passing the same batch as both sides forms every unordered pair exactly once.

        `softplus(-(l_i - l_j))` is the smooth stand-in for "is i above j": zero when i leads by a lot,
        linear in the gap when it trails. That is the surrogate an AUC's step function needs to become
        differentiable, and it is the same one the RankNet family uses.
        """
        gap = above.logits.unsqueeze(1) - below.logits.unsqueeze(0)
        separation = (above.scores.unsqueeze(1) - below.scores.unsqueeze(0)).clamp(min=0.0)
        confidence = above.weights.unsqueeze(1) * below.weights.unsqueeze(0)
        pair_weight = confidence * separation
        if emphasis is not None:
            pair_weight = pair_weight * emphasis
        misordered = pair_weight * nn.functional.softplus(-gap)
        return misordered.sum(), pair_weight.sum()


class DistilledObjective(RankedObjective):
    """The ranking objective, plus the same pairwise term judged against an ENSEMBLE'S ordering.

    WHY A RANKING TERM AND NOT THE KL THAT DISTILLATION USUALLY MEANS. Standard knowledge distillation
    matches temperature-softened probabilities, which is a CALIBRATION target. This competition scores
    twelve rank-based AUCs, and `inference/ensemble.py` states the consequence plainly: calibration is
    worth nothing here. A KL would spend the student's capacity reproducing probability values the
    metric discards, and temperature — being monotone — cannot change the target's order at all, so the
    one knob the technique is usually tuned on is provably inert against this metric.

    WHAT THE TEACHER ADDS THAT THE LABELS CANNOT. `pairs` weights a pair by `relu(s_i - s_j)`, so a pair
    teaches nothing when the two studies carry the SAME reading. The reader emits five levels, so under
    labels alone the pair mass collapses onto the level boundaries: a finding present in 7% of studies
    leaves roughly 87% of its pairs unordered, because both studies sit at the bottom level. The teacher
    is continuous and orders all of them, including the two-negatives and two-positives comparisons a
    five-level target cannot express. That is more supervision over the SAME corpus — which is the point,
    because the corpus is where this project is bound and no additional corpus has a usable licence.

    So the gain is predicted to CONCENTRATE on the rarest findings, by roughly six-fold across the panel,
    and a uniform lift across the twelve would refute the mechanism claimed here rather than confirm it.

    THE TEACHER MUST NOT HAVE TRAINED ON THESE STUDIES. Six of the seven champion members were fitted
    with `--fold -1`, so their readings of a training study are memorised rather than inferred, and
    distilling on those would teach the student the teacher's memorisation and score it on a holdout
    that flatters both. The fold-0 twins are the honest teachers; enforcing that is the caller's job,
    and it is the single assumption this class cannot check for itself.
    """

    def __init__(
        self,
        weight: float = DISTILL_WEIGHT,
        *,
        rank_weight: float = 0.0,
        normalise_by_count: bool = False,
        per_label: bool = False,
        structure: LabelStructure | None = None,
    ) -> None:
        super().__init__(rank_weight, normalise_by_count=normalise_by_count, per_label=per_label, structure=structure)
        self.distill_weight = weight

    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """Everything the ranking objective descends, plus the teacher's ordering at its own strength.

        The teacher is handed the FOCUSED batch, for the reason `RankedObjective.__call__` gives about
        its own term: the focus mask is idempotent, so both layers may apply it without coordinating.
        """
        focused = self.structure.focused(batch)
        return super().__call__(focused) + self.distill_weight * self.imitation(focused)

    @override
    def settings(self) -> dict[str, object]:
        """The ranking account plus the distillation term's real strength, not the flag that asked."""
        return {**super().settings(), DISTILL_TERM: self.distill_weight}

    def imitation(self, batch: Predicted) -> Float[Tensor, ""]:
        """How much of the teacher's ordering this batch's logits get the wrong way round.

        The whole term is the EXISTING `pairs` machinery pointed at a different target: the student's
        own logits judged against the teacher's scores. Nothing about the pair algebra changes, which is
        the reason to express it this way — the surrogate, the directionality and the self-pair
        cancellation are already argued and already tested where they are written.

        Pair mass is the FOCUS MASK rather than the label weights. Reusing the weights would import the
        reports' confidence into a term that is not judging the reports, and would silence the teacher
        exactly where the reports are least certain — which is where a teacher is worth the most. Taking
        `weights > 0` keeps a focused arm narrow, since focus is what zeroes a column, while leaving
        every surviving cell counted at full strength.

        RAISES when the batch carries no teacher, rather than contributing zero. A distilling arm whose
        targets never arrived would otherwise descend plain cross-entropy under this class's name and
        report a number that reads as a refutation of the mechanism — the exact failure `RANK_TERM`'s
        docstring was written about, and one this repo has already paid for in other flags.
        """
        if batch.teacher is None:
            message = (
                "a distilling objective was handed a batch with no teacher readings, so the term it "
                "exists for would contribute nothing — this is a plumbing failure, not an empty batch"
            )
            raise ValueError(message)
        taught = Predicted(
            logits=batch.logits,
            scores=batch.teacher,
            weights=(batch.weights > 0).to(batch.weights.dtype),
        )
        misordered, pair_weight = self.pairs(taught, taught)
        return misordered / pair_weight.clamp(min=PAIR_FLOOR)


class DecorrelatedObjective(RankedObjective):
    """The ranking objective, with its pairs re-weighted toward the ones the ENSEMBLE gets wrong.

    THE SAME LOADER AS DISTILLATION, AND THE OPPOSITE INTENT. `DistilledObjective` reads the pool's
    readings and pulls the student TOWARD that ordering, which by construction makes a member that is
    near the pool. This one reads the same array and spends the student's capacity where the pool's
    ordering FAILS. Both are the teacher file, the same batch alignment, the same pair algebra; only
    what the teacher is used for differs.

    WHY THIS AND NOT A CORRELATION PENALTY. Negative correlation learning adds a term rewarding a member
    for departing from the ensemble mean. Two things make that the wrong shape here. It is defined on
    VALUES, and this metric is twelve rank statistics that read no value — a member can decorrelate its
    scores completely without moving a single pair. And it is unbounded: nothing in a departure term
    says the departure should be CORRECT, so its whole job is done by the accuracy term holding it back,
    which is the tuning problem in the literature and a poor bet on a corpus this size.

    So the departure is anchored to the LABELS instead. Every pair this term pays for is a pair the
    reports order, judged in the direction the reports order it; the teacher only says which of those
    pairs to care about most. A member trained this way cannot drift into being wrong-but-different,
    because being different is only ever rewarded through being right.

    WHAT IT IS FOR, STATED SO IT CAN FAIL. `ensemble error = mean member error - ambiguity`. Every seat
    this project has bought or failed to buy moved exactly one of those terms: a distant trunk raised
    ambiguity and member error together (+0.0002), a swapped trunk left ambiguity flat (-0.0007). This
    term is the attempt to move both at once, so the prediction is specific — the member's SOLO score
    should land within noise of its control (it is still descending the labels), while its nearest
    seated neighbour drops. A member that comes back distant AND weak is the vjepa quadrant again and
    refutes this construction rather than the operating point.
    """

    def __init__(
        self,
        weight: float = DISSENT_WEIGHT,
        *,
        normalise_by_count: bool = False,
        per_label: bool = False,
        structure: LabelStructure | None = None,
    ) -> None:
        # The base ranking term is left OFF and not threaded through. `dissent` is that same term with a
        # re-weighted mass, so running both would descend the label ordering twice at two strengths and
        # neither number afterwards would say which of them moved the member.
        super().__init__(0.0, normalise_by_count=normalise_by_count, per_label=per_label, structure=structure)
        self.dissent_weight = weight

    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """Cross-entropy and the side term, plus the emphasised ranking term at its own strength."""
        focused = self.structure.focused(batch)
        return super().__call__(focused) + self.dissent_weight * self.dissent(focused)

    @override
    def settings(self) -> dict[str, object]:
        """The ranking account plus this term's real strength, not the flag that asked for it."""
        return {**super().settings(), DISSENT_TERM: self.dissent_weight}

    def dissent(self, batch: Predicted) -> Float[Tensor, ""]:
        """The label ranking term, with each pair's mass scaled by how badly the teacher orders it.

        The emphasis is `softplus(-(t_i - t_j))` — the SAME surrogate `pairs` uses for the student, read
        against the teacher's scores. It is near zero on a pair the pool orders confidently right, and
        grows linearly in the gap on one it orders confidently wrong. Both sums carry it, so the term is
        a re-weighted average and a teacher that separates nothing leaves `ranking` untouched, bit for
        bit — which is the equality that says the emphasis and not some rescaling is doing the work.

        The teacher is DETACHED. It is a fixed array read off disk, and letting the emphasis carry a
        gradient would be asking the student to make the pool look wrong, which is the unanchored term
        this class exists to avoid.

        RAISES on a batch with no teacher, for the reason `DistilledObjective.imitation` gives: a member
        whose readings never arrived would descend the plain ranking term under this class's name and
        report a null that reads as a refutation of the mechanism.
        """
        if batch.teacher is None:
            message = (
                "a decorrelating objective was handed a batch with no teacher readings, so it would "
                "descend the plain ranking term under its own name — a plumbing failure, not an "
                "empty batch"
            )
            raise ValueError(message)
        teacher = batch.teacher.detach()
        teacher_gap = teacher.unsqueeze(1) - teacher.unsqueeze(0)
        emphasis = nn.functional.softplus(-teacher_gap)
        misordered, pair_weight = self.pairs(batch, batch, emphasis)
        return misordered / pair_weight.clamp(min=PAIR_FLOOR)
