"""The terms an `Objective` sums, each one a `Term` strategy with its own mechanism and its own strength.

- `Ranking` — pays only for pairs ordered the wrong way.
- `Distillation` — that term judged against an ensemble's ordering.
- `Dissent` — that term, with its pairs re-weighted toward the ones an ensemble orders WRONG. The teacher
  file of the one above, read for the opposite purpose.
- `Margin` — the AUC-margin surrogate that won CheXpert, which REPLACES cross-entropy as the base rather
  than adding to it, on the argument that the calibration part of the answer is the part the scoreboard
  throws away.
- `Contrast` — per-finding supervised contrast over what the head READ, the one term not a function of
  the logits.

The ranking term is written for OUR supervision, not copied: our targets are weighted confidences in
[0, 1] rather than binary labels, so a pair contributes in proportion to how much the reports actually
separate the two studies and how confident both of those readings were.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Self, override

import torch
from jaxtyping import Float, Shaped
from torch import Tensor, nn

from kneemri.training.grades import Predicted

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

AUC_MARGIN = 1.0
"""How far apart the two classes' mean scores must be pulled before the margin term stops paying.

The scores it measures are sigmoid probabilities, so the whole available range is 1.0 and this default
asks for complete separation — which is the value Yuan et al. use and the one LibAUC ships. It is a
width, not a strength: `0.0` means the margin objective is OFF, and any positive value selects it.
"""

POSITIVE_AT = 0.5
"""Where a graded reading becomes a positive, which MUST be where the scorer cuts.

`LabelScores` binarises the holdout at its own `MIDPOINT` and the twelve AUCs are computed against
that. A margin objective cut anywhere else optimises a different partition than the one being scored,
silently and with no number afterwards that would say so — `test_the_margin_cuts_where_the_scorer_cuts`
is what keeps the two equal.
"""

CONTRAST_WEIGHT = 1.0
"""How hard the supervised-contrastive term pulls relative to cross-entropy when it is enabled at all.

Parity, for `RANK_WEIGHT`'s reason: both terms are means over their own populations and both sit near
`log(b)` ~ 2.5 versus cross-entropy's ~0.7 at initialisation. That is not equal in magnitude, which is
the honest statement — the contrastive term starts about three times larger and falls fastest, so this
default is the neutral START and not a claim that the two are balanced. Sweeping it before the term has
shown any signal would be measuring a method that has not yet worked.
"""

CONTRAST_TEMPERATURE = 0.1
"""The softmax temperature the similarities are read at — Khosla et al.'s published value, not ours.

Their ablation puts the optimum near 0.1 across ImageNet and CIFAR, and it is the value the reference
implementation ships. Taken as published rather than tuned, because a temperature swept on 4,407 studies
with twelve batch members would be fitting the sweep, not the temperature.
"""

EARLY_FLOOR = 1e-4
"""How close to certain a probability or an agreement may come before the log reads it — Liu et al.'s
clamp, which keeps `log(1 - agreement)` finite when a study has been read identically every epoch."""

RANK_TERM = "rank_weight"
EARLY_TERM = "early_weight"
DISTILL_TERM = "distill_weight"
FEATURE_TERM = "feature_weight"
DISSENT_TERM = "dissent_weight"
CONTRAST_TERM = "contrast_weight"
MARGIN_TERM = "auc_margin"
PAIR_POOL = "pair_memory"
"""The vocabulary an objective answers `settings()` in, so a run can check its loss switches landed.

Every combination of terms is a working objective: a `--rank-weight` that never reached its term trains
plain cross-entropy under a ranking arm's name and reports a number that reads as a refutation of the
mechanism.
"""

PAIR_MEMORY = 8
"""How many past steps' readouts a `Remembered` term keeps, when it is enabled at all.

Set by the two quantities that actually trade here. UPWARD: pairs. A batch of 12 forms 66 unordered
pairs per label and most of them separate by nothing, while eight remembered steps put 96 more studies
on the other side of every comparison — about seventeen times the pair mass, which is the whole point.
DOWNWARD: staleness. A remembered logit was produced by weights that have since taken up to eight
steps, and a pair judged against a model that no longer exists is noise. Eight steps is under 3% of an
epoch's 291 here, and the encoder moves at 8e-6, so the drift over the window is far below the gap
`softplus` is measuring.

It is a flag with an explicit default, not a swept hyper-parameter, for the reason RANK_WEIGHT gives.
"""


class Term(ABC):
    """One scalar the objective descends, read off the focused batch and summed at its own strength.

    A strategy rather than a subclass layer. The terms used to be a chain five constructors deep, each
    one the last plus a term, so composing two of them meant choosing which one's parent the other got
    to be — and the builder answered by returning whichever it checked first and dropping the rest.
    """

    key: ClassVar[str] = ""

    def __init__(self, weight: float) -> None:
        self.weight = weight

    @abstractmethod
    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """This term's scalar for one batch, before the objective scales it by `weight`."""

    def graded(self, batch: Predicted, spread: Callable[[Predicted], Predicted]) -> Float[Tensor, ""]:
        """This term read as a BASE: on the batch the grade policy restates, spread as the last step."""
        return self(spread(batch))

    def settings(self) -> dict[str, object]:
        """The term's real strength under its own key, not the flag that asked for it."""
        return {self.key: self.weight}

    def attached(self, device: torch.device) -> list[nn.Parameter]:
        """Parameters this term OWNS and the optimiser must step — none, for every term but one.

        A term holding a learned projection would otherwise sit in no parameter group at all: the
        optimiser is built from `StudyModel.parameter_groups`, which collects by EXCLUSION from the
        backbone and cannot see a tensor the loss holds. That is knee-7gyw's defect exactly, and it is
        silent — the projection stays at its initialisation and the arm reads as a fact about the term.
        """
        del device
        return []

    def arrived(self, value: Shaped[Tensor, "..."] | None, what: str) -> Shaped[Tensor, "..."]:
        """`value`, or a refusal naming what never arrived — never a zero contribution.

        A term whose inputs never reached it would otherwise descend the plain recipe under the arm's
        name and report a number that reads as a refutation of the mechanism — the failure `RANK_TERM`'s
        docstring was written about, and one this repo has already paid for in other flags.
        """
        if value is None:
            message = (
                f"{self.key} was handed a batch with no {what}, so the term it exists for would contribute "
                "nothing — this is a plumbing failure, not an empty batch"
            )
            raise ValueError(message)
        return value


@dataclass(frozen=True)
class Widening:
    """The live batch beside the pooled memory it is widened by — the pair every `Pooled` term reads.

    A value rather than two staticmethods on `Pooled` that both threaded `(batch, remembered)`: the pair
    is one thing, and its order is part of what it means.
    """

    batch: Predicted
    remembered: Predicted

    def readout(self) -> Predicted:
        """The live batch followed by the pooled memory, as one readout — `[live, remembered]`, never reversed.

        Every anchor has to be its own first key, so the diagonal of the leading block is the only entry a
        self-exclusion masks out. `teacher` is dropped because no pooled term reads one and a half-carried
        teacher is worse than none.
        """
        return Predicted(
            logits=torch.cat([self.batch.logits, self.remembered.logits]),
            scores=torch.cat([self.batch.scores, self.remembered.scores]),
            weights=torch.cat([self.batch.weights, self.remembered.weights]),
            embedding=self.embedding(),
        )

    def embedding(self) -> Float[Tensor, "b o h"] | None:
        """Both sides' contexts as one, or None when either side has none — which a reader then REFUSES.

        None rather than the live half alone: silently narrowing back to the batch would return a
        plausible number about the term the memory was built to widen.
        """
        if self.batch.embedding is None or self.remembered.embedding is None:
            return None
        return torch.cat([self.batch.embedding, self.remembered.embedding])


class Pooled(Term):
    """A term that can be read over a WIDER population than one batch — what `Remembered` widens.

    The live batch's gradient is the only one that flows; the remembered side is a fixed target, exactly
    as in a contrastive memory queue.
    """

    @abstractmethod
    def across(self, batch: Predicted, remembered: Predicted) -> Float[Tensor, ""]:
        """This term with the live batch compared against itself AND the remembered steps."""


class CrossEntropy(Term):
    """Confidence-weighted cross-entropy, normalised either by total weight or by element count — the default base.

    BY WEIGHT (ours): a batch carrying little confidence does not take a full-sized step on almost no
    evidence. BY COUNT (the public baseline's `.mean()`): the step size is a fixed scale that does not
    depend on how confidently the batch happened to be labelled.

    The two are not interchangeable in magnitude. With most weights below one, dividing by their sum
    makes our gradients roughly twice theirs for the same batch — which interacts with the learning-rate
    schedule, so the pair has to be measured together rather than one at a time.
    """

    def __init__(self, *, per_label: bool = False, normalise_by_count: bool = False) -> None:
        super().__init__(1.0)
        self.per_label = per_label
        self.normalise_by_count = normalise_by_count

    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """The weighted BCE over a batch the grade policy has already restated."""
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

    @override
    def settings(self) -> dict[str, object]:
        """Nothing beyond the base account, which already carries both pooling switches."""
        return {}


_EXCLUDED = -1e4
"""What a similarity a study forms with ITSELF becomes before the log-sum-exp: negligible after the
exponential, finite in fp16. A hard `-inf` would produce a NaN gradient rather than a zero one."""


@dataclass(frozen=True)
class PairMass:
    """The misordering mass asking one side to outrank the other, and the pair mass it is out of.

    Kept unreduced as a (numerator, denominator) pair rather than as their ratio, because a caller pooling
    several populations of pairs must add the two sums BEFORE dividing. Averaging ratios instead would
    give a handful of within-batch pairs the same say as a hundred remembered ones, which is the opposite
    of the point.
    """

    misordered: Float[Tensor, ""]
    total: Float[Tensor, ""]

    def __add__(self, other: "PairMass") -> "PairMass":
        """Both populations as one: numerators and denominators summed separately."""
        return PairMass(self.misordered + other.misordered, self.total + other.total)

    def ratio(self) -> Float[Tensor, ""]:
        """The share of the pair mass ordered the wrong way — the term itself."""
        return self.misordered / self.total.clamp(min=PAIR_FLOOR)

    @classmethod
    def between(
        cls,
        above: Predicted,
        below: Predicted,
        emphasis: Float[Tensor, "b b o"] | None = None,
    ) -> Self:
        """The mass asking `above` to outrank `below`, every ordered pair within a label formed at once.

        A pair counts in proportion to `w_i * w_j * relu(s_i - s_j)`: it is directional (only pairs the
        reports say should go one way are asked to go that way, and the reverse pair carries the mirror
        term), it vanishes when the two readings agree, and it vanishes when either reading is a guess.
        Pairs a study forms with itself have `s_i - s_j = 0` and so drop out without a mask — which is
        also why passing the same batch as both sides forms every unordered pair exactly once.

        `emphasis` scales the mass of each pair INDIVIDUALLY, which is the one thing a caller cannot
        express by editing the two `Predicted`s: `weights` are per study, and "how much does this
        COMPARISON matter" is per pair. It multiplies BOTH sums, so a uniform emphasis, whatever its size,
        is exactly no emphasis.

        `softplus(-(l_i - l_j))` is the smooth stand-in for "is i above j": zero when i leads by a lot,
        linear in the gap when it trails — the surrogate the RankNet family uses.
        """
        gap = above.logits.unsqueeze(1) - below.logits.unsqueeze(0)
        separation = (above.scores.unsqueeze(1) - below.scores.unsqueeze(0)).clamp(min=0.0)
        confidence = above.weights.unsqueeze(1) * below.weights.unsqueeze(0)
        pair_weight = confidence * separation
        if emphasis is not None:
            pair_weight = pair_weight * emphasis
        misordered = pair_weight * nn.functional.softplus(-gap)
        return cls(misordered.sum(), pair_weight.sum())


class Ranking(Pooled):
    """A soft, confidence-weighted count of the pairs a batch ranks the wrong way round.

    Kept as a term ON TOP OF cross-entropy rather than as a ranking loss alone. A pure ranking objective
    fixes nothing about the scale of the logits, which leaves the sigmoid free to drift somewhere flat
    and the gradient with it; and the twelve labels are then coupled only through the encoder, so a rare
    finding with few separating pairs in a batch would contribute almost nothing to any step.

    Its pair mass is `weights_i * weights_j`, so a column the focus policy zeroed forms no pairs and
    contributes to neither sum — which is what makes one mask on the weights enough to narrow it.
    """

    key = RANK_TERM

    def __init__(self, weight: float = RANK_WEIGHT) -> None:
        super().__init__(weight)

    @override
    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """The within-batch pairs, as one broadcast over (b, b, o) rather than a loop over labels."""
        return PairMass.between(batch, batch).ratio()

    @override
    def across(self, batch: Predicted, remembered: Predicted) -> Float[Tensor, ""]:
        """The within-batch pairs and the cross-step ones, pooled as one population.

        BOTH DIRECTIONS ARE FORMED. The mass is directional, and asking only whether the live batch
        outranks the remembered one would silently drop every pair a remembered study should win — half
        the comparisons, and the half biased toward whichever readings happen to be strong right now.
        """
        pooled = (
            PairMass.between(batch, batch) + PairMass.between(batch, remembered) + PairMass.between(remembered, batch)
        )
        return pooled.ratio()


class Distillation(Term):
    """The ranking term judged against an ENSEMBLE'S ordering rather than the reports'.

    WHY A RANKING TERM AND NOT THE KL THAT DISTILLATION USUALLY MEANS. Standard knowledge distillation
    matches temperature-softened probabilities, which is a CALIBRATION target. This competition scores
    twelve rank-based AUCs, and `inference/ensemble.py` states the consequence plainly: calibration is
    worth nothing here. A KL would spend the student's capacity reproducing probability values the
    metric discards, and temperature — being monotone — cannot change the target's order at all, so the
    one knob the technique is usually tuned on is provably inert against this metric.

    WHAT THE TEACHER ADDS THAT THE LABELS CANNOT. A pair teaches nothing when the two studies carry the
    SAME reading. The reader emits five levels, so under labels alone the pair mass collapses onto the
    level boundaries: a finding present in 7% of studies leaves roughly 87% of its pairs unordered,
    because both studies sit at the bottom level. The teacher is continuous and orders all of them. That
    is more supervision over the SAME corpus — which is the point, because the corpus is where this
    project is bound and no additional corpus has a usable licence.

    So the gain is predicted to CONCENTRATE on the rarest findings, by roughly six-fold across the panel,
    and a uniform lift across the twelve would refute the mechanism claimed here rather than confirm it.

    THE TEACHER MUST NOT HAVE TRAINED ON THESE STUDIES. A `--fold -1` member's readings of a training
    study are memorised rather than inferred, and distilling on those would teach the student the
    teacher's memorisation and score it on a holdout that flatters both. The fold-0 twins are the honest
    teachers; enforcing that is the caller's job, and it is the single assumption this term cannot check.
    """

    key = DISTILL_TERM

    def __init__(self, weight: float = DISTILL_WEIGHT) -> None:
        super().__init__(weight)

    @override
    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """How much of the teacher's ordering this batch's logits get the wrong way round.

        Pair mass is the FOCUS MASK rather than the label weights. Reusing the weights would import the
        reports' confidence into a term that is not judging the reports, and would silence the teacher
        exactly where the reports are least certain — which is where a teacher is worth the most. Taking
        `weights > 0` keeps a focused arm narrow, since focus is what zeroes a column, while leaving
        every surviving cell counted at full strength.
        """
        taught = Predicted(
            logits=batch.logits,
            scores=self.arrived(batch.teacher, "teacher readings"),
            weights=(batch.weights > 0).to(batch.weights.dtype),
        )
        return PairMass.between(taught, taught).ratio()


class Dissent(Term):
    """The label ranking term, with its pairs re-weighted toward the ones the ENSEMBLE gets wrong.

    THE SAME LOADER AS DISTILLATION, AND THE OPPOSITE INTENT. `Distillation` pulls the student TOWARD the
    pool's ordering, which by construction makes a member that is near the pool. This spends the
    student's capacity where the pool's ordering FAILS.

    WHY THIS AND NOT A CORRELATION PENALTY. Negative correlation learning rewards a member for departing
    from the ensemble mean. It is defined on VALUES, and this metric is twelve rank statistics that read
    no value; and it is unbounded — nothing says the departure should be CORRECT. So the departure is
    anchored to the LABELS instead: every pair this term pays for is a pair the reports order, judged in
    the direction the reports order it, and the teacher only says which of those pairs to care about most.

    NEVER BESIDE `Ranking`. This is that same term with a re-weighted mass, so running both would descend
    the label ordering twice at two strengths and neither number afterwards would say which moved the
    member — `ObjectiveChecks` refuses the pair.

    WHAT IT IS FOR, STATED SO IT CAN FAIL. `ensemble error = mean member error - ambiguity`. The
    prediction is specific — the member's SOLO score should land within noise of its control, while its
    nearest seated neighbour drops. A member that comes back distant AND weak refutes this construction.
    """

    key = DISSENT_TERM

    def __init__(self, weight: float = DISSENT_WEIGHT) -> None:
        super().__init__(weight)

    @override
    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """The label ranking term, with each pair's mass scaled by how badly the teacher orders it.

        The emphasis is `softplus(-(t_i - t_j))` — the SAME surrogate the pair mass uses for the student,
        read against the teacher's scores. Both sums carry it, so a teacher that separates nothing leaves
        `Ranking` untouched, bit for bit.

        The teacher is DETACHED: letting the emphasis carry a gradient would ask the student to make the
        pool look wrong, which is the unanchored term this exists to avoid.
        """
        teacher = self.arrived(batch.teacher, "teacher readings").detach()
        emphasis = nn.functional.softplus(-(teacher.unsqueeze(1) - teacher.unsqueeze(0)))
        return PairMass.between(batch, batch, emphasis).ratio()


class EarlyLearning(Term):
    """Liu et al.'s early-learning regulariser: each study held near what the model said about it EARLIER.

    WHY A LOSS AND NOT ANOTHER TARGET. Every label-noise arm this project ran changed the targets —
    calibration, relabelling, pruning, distillation, transduction. None changed what the loss does with a
    target it cannot trust. The reader over-calls: on the 58 radiologist-read studies 108 of its 140 wrong
    cells are false positives, concentrated on mild or hedged findings that are image-plausible, so a
    network can memorise them late in a run and then rank them high on a radiologist-read test.

    THE MECHANISM (Liu et al., NeurIPS 2020, eq. 6, written for twelve sigmoids rather than one softmax).
    A network fits the clean part of a noisy table first and the noise afterwards. The target here is a
    running average of the network's own past readings of each study, so it carries the early, clean fit;
    `log(1 - agreement)` falls as the current reading moves toward it, which pulls against a late jump onto
    a noisy label. For one sigmoid the inner product over the two outcomes is
    `p * positive + (1 - p) * negative`, where the two components are the averaged past `p` and `1 - p`.

    UNNORMALISED, AS PUBLISHED. Both components start at zero, so a study's first visit contributes exactly
    nothing and the term's strength grows with `1 - beta ** visits`: an epoch-scale warm-up the paper gets
    from its zero initialisation and that normalising the average would throw away.

    The pair mass is the FOCUS MASK, for `Distillation`'s reason: this term does not judge the reports, so
    their confidence has no say in it.
    """

    key = EARLY_TERM

    @override
    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """The mean of `log(1 - agreement)` over the cells the focus policy kept."""
        remembered = self.arrived(batch.early, "early-learning targets")
        probability = batch.logits.sigmoid().clamp(EARLY_FLOOR, 1.0 - EARLY_FLOOR)
        agreement = probability * remembered[..., 0] + (1.0 - probability) * remembered[..., 1]
        kept = (batch.weights > 0).to(probability.dtype)
        pulled = torch.log((1.0 - agreement).clamp(min=EARLY_FLOOR)) * kept
        return pulled.sum() / kept.sum().clamp(min=1.0)


class Margin(Pooled):
    """The AUC-margin surrogate — a BASE, the one term here that replaces cross-entropy rather than adding.

    Its weight is the margin WIDTH, not a strength: the base is summed unscaled.

    PRIOR ART, not an invention here. Deep AUC Maximization with the AUC-margin surrogate won the
    Stanford CheXpert competition in 2020 — multi-label, class-imbalanced, medical imaging, scored on
    mean AUROC. Yuan et al., *Large-scale Robust Deep AUC Maximization* (arXiv 2012.03173, ICCV 2021);
    reference implementation LibAUC.

    WRITTEN IN CLOSED FORM. The published loss is a min-max problem over per-label scalars `(a, b)` and a
    dual `alpha`. Both inner problems are quadratic and solved exactly, so substituting them (the
    envelope theorem leaves the outer gradient unchanged) collapses it to

        var[p | positive] + var[p | negative] + relu(margin - (mean_pos - mean_neg))^2

    — no extra parameters and no second optimiser.

    THE READING BINARISES WHERE THE SCORER DOES, at `POSITIVE_AT`, keeping the graded confidence only as
    the per-study weight.

    THE KNOWN WEAKNESS. A label needs both a positive and a negative in the batch to form this term, and
    Fracture is positive in ~7% of twelve-study batches, which biases the panel AGAINST the rare labels.
    `Remembered` is the fix.
    """

    key = MARGIN_TERM

    def __init__(self, weight: float = AUC_MARGIN) -> None:
        super().__init__(weight)

    @override
    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """Within-class spread plus the shortfall between the two class means, averaged over live labels.

        `normalise_by_count` and `per_label` do not reach this term: it is already one number per label,
        and it already gives each of the twelve an equal say.

        A label with no positive or no negative in the batch forms no comparison and is DROPPED from the
        mean rather than contributing a zero, which would read as a label already separated.
        """
        probability = batch.logits.sigmoid()
        positive = (batch.scores > POSITIVE_AT).to(probability.dtype) * batch.weights
        negative = (batch.scores <= POSITIVE_AT).to(probability.dtype) * batch.weights
        positive_mass, negative_mass = positive.sum(dim=0), negative.sum(dim=0)
        live = (positive_mass > 0.0) & (negative_mass > 0.0)
        if not bool(live.any()):
            return probability.sum() * 0.0
        positive_mean = (positive * probability).sum(dim=0) / positive_mass.clamp(min=PAIR_FLOOR)
        negative_mean = (negative * probability).sum(dim=0) / negative_mass.clamp(min=PAIR_FLOOR)
        spread = (positive * (probability - positive_mean) ** 2).sum(dim=0) / positive_mass.clamp(min=PAIR_FLOOR) + (
            negative * (probability - negative_mean) ** 2
        ).sum(dim=0) / negative_mass.clamp(min=PAIR_FLOOR)
        shortfall = (self.weight - (positive_mean - negative_mean)).clamp(min=0.0)
        return (spread + shortfall**2)[live].mean()

    @override
    def across(self, batch: Predicted, remembered: Predicted) -> Float[Tensor, ""]:
        """The same surrogate with the class means and spreads taken over the remembered steps too."""
        return self(Widening(batch, remembered).readout())


class Contrast(Pooled):
    """Khosla et al.'s SupCon, computed per finding over the head's own per-finding contexts.

    THE ONE TERM HERE THAT IS NOT A FUNCTION OF THE LOGITS. Two studies can share the twelve logits while
    the head read them for entirely different reasons. This is written against `Readout.context` — the
    vector before the collapse — so it constrains the REPRESENTATION and not the decision.

    PER FINDING AND NOT PER STUDY. A study is typically torn in one compartment and intact in eleven other
    respects, so one embedding per study would ask a single vector to be near the torn cluster and the
    intact cluster at once. A study is an anchor twelve times over with a different partition each time.

    WHY THIS IS NOT THE FLAT SPATIAL-POOLING RESULT AGAIN. Every locality lever measured here is an
    INTENSITY STATISTIC over patch tokens or slices, and all are flat to negative. This answers "what does
    the difference between torn and intact look like", which no pooling operator expresses — knee-ek5s.

    THE SUPERVISION IS BINARISED AT THE SCORER'S CUT, `POSITIVE_AT`; the graded confidence survives as the
    anchor's WEIGHT.

    THE KNOWN WEAKNESS is `Margin`'s: a finding positive in 7% of the corpus supplies an anchor with a
    positive partner in a minority of twelve-study steps. `Remembered` is the fix.
    """

    key = CONTRAST_TERM

    def __init__(self, weight: float = CONTRAST_WEIGHT, *, temperature: float = CONTRAST_TEMPERATURE) -> None:
        super().__init__(weight)
        self.temperature = temperature

    @override
    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """SupCon with the batch as both anchors and keys."""
        return self.supcon(batch, batch)

    @override
    def across(self, batch: Predicted, remembered: Predicted) -> Float[Tensor, ""]:
        """SupCon with the live batch as anchors and the live batch plus the remembered steps as keys."""
        return self.supcon(batch, Widening(batch, remembered).readout())

    def supcon(self, anchors: Predicted, keys: Predicted) -> Float[Tensor, ""]:
        """SupCon with the anchors and the compared population named separately.

        THE CONTRACT, AND IT IS WHAT THE SELF-EXCLUSION RESTS ON: `keys` BEGINS WITH `anchors`. Every
        anchor is its own first key, so the diagonal of the leading block is the study compared with
        itself and is the only entry masked out.

        An anchor with no positive OTHER than itself forms no term and is dropped rather than contributing
        a zero, which would read as a finding already separated.
        """
        z = nn.functional.normalize(self.arrived(anchors.embedding, "embedding").float(), dim=-1).permute(1, 0, 2)
        against = nn.functional.normalize(self.arrived(keys.embedding, "embedding").float(), dim=-1).permute(1, 0, 2)
        similarity = torch.einsum("obh,och->obc", z, against) / self.temperature
        studies, compared = z.shape[1], against.shape[1]
        other = ~torch.eye(studies, compared, dtype=torch.bool, device=z.device)
        positive = (anchors.scores > POSITIVE_AT).T
        opposite = (keys.scores > POSITIVE_AT).T
        # A PAIR IS POSITIVE WHEN BOTH STUDIES FALL ON THE SAME SIDE OF THE SCORER'S CUT — intact with
        # intact as much as torn with torn. Contrasting only the positives against everything would ask
        # the healthy studies to be mutually unlike, which the anatomy says they are not.
        pairs = (positive.unsqueeze(2) == opposite.unsqueeze(1)) & other.unsqueeze(0)
        denominator = similarity.masked_fill(~other.unsqueeze(0), _EXCLUDED).logsumexp(dim=2)
        share = pairs.sum(dim=2)
        live = share > 0
        if not bool(live.any()):
            return z.sum() * 0.0
        agreement = (similarity * pairs).sum(dim=2) / share.clamp(min=1)
        # THE ANCHOR'S OWN CONFIDENCE, not the pair's: `pairs` already averages over the partners.
        mass = (anchors.weights.T * live).clamp(min=0.0)
        loss = (denominator - agreement) * mass
        return loss.sum() / mass.sum().clamp(min=PAIR_FLOOR)
