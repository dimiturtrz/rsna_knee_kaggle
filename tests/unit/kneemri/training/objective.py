import pytest
import torch

from core.findings import COMPARTMENT_PAIRS, TARGETS
from kneemri.training.grades import GRADE_LADDER, OrdinalGrades, Predicted
from kneemri.training.label_focus import FOCUS, LabelFocus
from kneemri.training.objective import (
    COUNT_NORMALISED,
    DISTILL_TERM,
    PAIR_POOL,
    PER_LABEL,
    RANK_TERM,
    DistilledObjective,
    LabelStructure,
    Objective,
    RankedObjective,
)
from kneemri.training.side_ranking import SIDE_TERM, PairedSides

SEED = 2026
STUDIES = 6
LABELS = 3


def _batch(order: list[float]) -> Predicted:
    """One label's worth of logits laid out in `order`, with binary targets and full confidence."""
    logits = torch.tensor(order).reshape(-1, 1)
    scores = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
    return Predicted(logits, scores, torch.ones_like(scores))


def test_objective_settings():
    """The plain objective's own account of its loss switches, which a run checks before it starts.

    It reports the ranking term OFF rather than omitting the key: the question is whether a ranking arm
    got a ranking objective at all, and a missing key cannot answer it.
    """
    assert Objective(per_label=True).settings() == {
        RANK_TERM: 0.0,
        DISTILL_TERM: 0.0,
        PAIR_POOL: 0,
        PER_LABEL: True,
        COUNT_NORMALISED: False,
        GRADE_LADDER: False,
        SIDE_TERM: 0.0,
        FOCUS: "",
    }


def test_ranked_objective_settings():
    """The ranked one reports the term's real strength, not the flag that asked for it."""
    assert RankedObjective(0.3, normalise_by_count=True).settings() == {
        RANK_TERM: 0.3,
        DISTILL_TERM: 0.0,
        PAIR_POOL: 0,
        PER_LABEL: False,
        COUNT_NORMALISED: True,
        GRADE_LADDER: False,
        SIDE_TERM: 0.0,
        FOCUS: "",
    }


def test_cross_entropy():
    """The weighted BCE term is finite, positive, and falls as the logits approach the targets."""
    scores = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    weights = torch.ones_like(scores)
    objective = Objective()

    wrong = objective.cross_entropy(Predicted(torch.tensor([[-4.0, 4.0], [4.0, -4.0]]), scores, weights))
    right = objective.cross_entropy(Predicted(torch.tensor([[4.0, -4.0], [-4.0, 4.0]]), scores, weights))
    assert right < wrong
    assert float(right) > 0.0


def test_cross_entropy_normalisation_modes_differ():
    """Dividing by total weight is not the same step as dividing by element count.

    With most weights below one the by-weight form gives the larger gradient for the same batch, which
    is why the two cannot be swapped without re-measuring the learning rate alongside.
    """
    logits = torch.zeros(4, 2)
    scores = torch.ones(4, 2)
    weights = torch.full((4, 2), 0.25)

    by_weight = Objective().cross_entropy(Predicted(logits, scores, weights))
    by_count = Objective(normalise_by_count=True).cross_entropy(Predicted(logits, scores, weights))
    assert not torch.isclose(by_weight, by_count)
    assert by_weight > by_count


def test_ranked_objective_ranking():
    """A batch ordered the right way costs near nothing; the same batch reversed costs a lot."""
    objective = RankedObjective()
    correct = objective.ranking(_batch([6.0, 5.0, 4.0, -4.0, -5.0, -6.0]))
    reversed_order = objective.ranking(_batch([-6.0, -5.0, -4.0, 4.0, 5.0, 6.0]))

    assert float(correct) < 0.01
    assert float(reversed_order) > 1.0


def test_ranking_is_blind_to_a_shift_that_leaves_the_order_alone():
    """The whole point: the term reads ORDER, so adding a constant to every logit changes nothing.

    Cross-entropy on the same two batches differs a great deal, which is the calibration work the
    metric discards.
    """
    objective = RankedObjective()
    batch = _batch([6.0, 5.0, 4.0, -4.0, -5.0, -6.0])
    lifted = Predicted(batch.logits + 3.0, batch.scores, batch.weights)

    assert torch.isclose(objective.ranking(batch), objective.ranking(lifted), atol=1e-6)
    assert not torch.isclose(objective.cross_entropy(batch), objective.cross_entropy(lifted))


def test_ranking_ignores_pairs_the_reports_do_not_separate():
    """Two studies read the same way form no pair, so a batch with one reading costs nothing."""
    logits = torch.tensor([[3.0], [-3.0], [0.5]])
    same = torch.full((3, 1), 0.7)
    assert float(RankedObjective().ranking(Predicted(logits, same, torch.ones_like(same)))) == 0.0


def test_ranking_discounts_a_pair_resting_on_a_guess():
    """A pair is weighted by BOTH readings' confidence, so an unconfident study barely pulls."""
    objective = RankedObjective()
    batch = _batch([-6.0, -5.0, -4.0, 4.0, 5.0, 6.0])

    confident = objective.ranking(batch)
    unsure = objective.ranking(Predicted(batch.logits, batch.scores, batch.weights * 0.1))
    assert torch.isclose(unsure, confident), "the term is a weighted MEAN, so a uniform discount cancels"

    partial = batch.weights.clone()
    partial[0] = 0.0
    assert float(objective.ranking(Predicted(batch.logits, batch.scores, partial))) < float(confident)


def test_ranking_objective_adds_the_term_at_its_configured_strength():
    """The combined objective is exactly cross-entropy plus weight times the ranking term."""
    batch = _batch([-6.0, -5.0, -4.0, 4.0, 5.0, 6.0])
    objective = RankedObjective(0.5)

    expected = objective.cross_entropy(batch) + 0.5 * objective.ranking(batch)
    assert torch.isclose(objective(batch), expected)


def test_ranking_at_zero_strength_is_plain_cross_entropy():
    """Turning the term off must leave the measured default bit-for-bit unchanged."""
    batch = _batch([-6.0, -5.0, -4.0, 4.0, 5.0, 6.0])
    assert torch.equal(RankedObjective(0.0)(batch), Objective()(batch))


def test_ranking_carries_a_gradient_to_every_study_in_a_misordered_batch():
    """A loss that cannot reach the logits is not training anything — the failure a value test misses."""
    torch.manual_seed(SEED)
    logits = torch.randn(STUDIES, LABELS, requires_grad=True)
    scores = torch.randint(0, 2, (STUDIES, LABELS)).float()
    RankedObjective().ranking(Predicted(logits, scores, torch.ones_like(scores))).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0


def test_cross_entropy_per_label_gives_every_label_the_same_say():
    """Pooled, a label's share of the gradient tracks how OFTEN it is mentioned; the metric does not.

    Effusion is positive in 60% of studies and Fracture in 7%, so the pooled objective spends about ten
    times more on one than the other while the score averages the twelve equally. Normalising per column
    removes that, which is the only sense in which the loss can be said to match the metric.
    """
    common, rare = 0, 1
    logits = torch.zeros(20, 2)
    logits[:, common] = 3.0  # mentioned everywhere, and nearly right
    logits[0, rare] = -3.0  # mentioned once, and badly wrong
    scores = torch.ones(20, 2)
    weights = torch.zeros(20, 2)
    weights[:, common] = 1.0
    weights[0, rare] = 1.0

    pooled = float(Objective()(Predicted(logits, scores, weights)))
    balanced = float(Objective(per_label=True)(Predicted(logits, scores, weights)))
    assert balanced > pooled * 4, "the rare label must dominate once every column counts the same"

    # And with the columns equally represented the two agree, so this is a REWEIGHTING and not a rescale.
    even = torch.ones(20, 2)
    pooled_even = float(Objective()(Predicted(logits, scores, even)))
    balanced_even = float(Objective(per_label=True)(Predicted(logits, scores, even)))
    assert abs(pooled_even - balanced_even) < 1e-6, "equal representation, equal answer — a reweighting, not a rescale"


def test_pairs():
    """The (numerator, denominator) split must still be the ratio `ranking` used to return on its own."""
    batch = _batch([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    misordered, pair_weight = RankedObjective().pairs(batch, batch)

    assert float(pair_weight) > 0.0
    assert torch.allclose(misordered / pair_weight, RankedObjective().ranking(batch))


def test_ordinal_grades_leave_the_ranking_term_alone():
    """The ladder is a cross-entropy policy, and the ranking term still compares the twelve latents.

    THIS IS THE SEPARATION THE DESIGN RESTS ON: applied in `__call__` instead, a ranking arm would be
    ordering forty-eight threshold crossings against each other, which is a different experiment
    wearing this one's name.
    """
    batch = _batch([1.0, 0.5, 0.0, -0.5, -1.0, -1.5])
    plain, laddered = RankedObjective(), RankedObjective(structure=LabelStructure(grades=OrdinalGrades()))
    assert torch.isclose(plain.ranking(batch), laddered.ranking(batch))
    assert not torch.isclose(plain.cross_entropy(batch), laddered.cross_entropy(batch))


def _twelve() -> Predicted:
    """A batch wide enough for the side policy, whose columns are found by name in the twelve."""
    columns = {target: index for index, target in enumerate(TARGETS)}
    scores = torch.zeros(4, len(TARGETS))
    for medial, lateral in COMPARTMENT_PAIRS:
        scores[:2, columns[lateral]] = 1.0
        scores[2:, columns[medial]] = 1.0
    return Predicted(torch.zeros(4, len(TARGETS)), scores, torch.ones(4, len(TARGETS)))


def test_label_structure_settings():
    """All three held policies are reported, so a run can check any one of them landed."""
    assert LabelStructure().settings() == {GRADE_LADDER: False, SIDE_TERM: 0.0, FOCUS: ""}
    assert LabelStructure(grades=OrdinalGrades()).settings()[GRADE_LADDER] is True
    assert LabelStructure(sides=PairedSides(0.5)).settings()[SIDE_TERM] == 0.5
    assert LabelStructure(focus=LabelFocus(TARGETS[3])).settings()[FOCUS] == TARGETS[3]


def test_focused():
    """The structure delegates the mask, and the null policy hands the same object back.

    Identity rather than equality on the null case: every member shipped to date was fitted through
    this branch, and a mask of ones would multiply where the old runs did not.
    """
    batch = _twelve()

    assert LabelStructure().focused(batch) is batch
    assert float(LabelStructure(focus=LabelFocus(TARGETS[3])).focused(batch).weights.sum()) == float(
        batch.weights[:, 3].sum()
    )


def test_label_structure_spread():
    """The null grade policy hands the batch back untouched; the ladder restates it."""
    batch = _twelve()

    assert LabelStructure().spread(batch).logits.shape == batch.logits.shape
    assert LabelStructure(grades=OrdinalGrades()).spread(batch).logits.shape[-1] > batch.logits.shape[-1]


def test_label_structure_side_term():
    """A hard zero under the null policy, so an objective built without it descends the old scalar."""
    batch = _twelve()

    assert float(LabelStructure().side_term(batch)) == 0.0
    assert float(LabelStructure(sides=PairedSides(1.0)).side_term(batch)) > 0.0


def _taught(order: list[float], teacher: list[float]) -> Predicted:
    """One label's logits laid out in `order`, judged against a teacher that ranks them by `teacher`.

    The label scores are held IDENTICAL across every study on purpose. That is the equivalence class the
    whole mechanism rests on: a pair the reports do not separate contributes nothing to `ranking`, so any
    signal these batches carry can only have come from the teacher.
    """
    logits = torch.tensor(order).reshape(-1, 1)
    flat = torch.full((len(order), 1), 0.08)
    return Predicted(logits, flat, torch.ones_like(flat), teacher=torch.tensor(teacher).reshape(-1, 1))


def test_distilled_objective_settings():
    """The distilling objective's account carries BOTH terms, because both are live in the same run.

    The owner's design is one training with three exits, so a checkpoint has to record the distillation
    strength beside the ranking strength — a run that reports only one of them cannot afterwards say
    which term produced its number.
    """
    settings = DistilledObjective(0.5, rank_weight=0.25).settings()

    assert settings[DISTILL_TERM] == 0.5
    assert settings[RANK_TERM] == 0.25, "the ranking term is still reported, not shadowed by the new one"
    assert settings[PAIR_POOL] == 0
    assert settings[PER_LABEL] is False
    assert settings[COUNT_NORMALISED] is False


def test_imitation():
    """Agreeing with the teacher's order costs less than contradicting it, which is the whole term."""
    objective = DistilledObjective()
    teacher = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]

    agreeing = objective.imitation(_taught([3.0, 2.0, 1.0, -1.0, -2.0, -3.0], teacher))
    contradicting = objective.imitation(_taught([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], teacher))

    assert float(agreeing) < float(contradicting)
    assert float(agreeing) < 0.1, "a student that already ranks the teacher's way owes almost nothing"


def test_imitation_orders_pairs_the_labels_cannot():
    """THE MECHANISM, and the reason this term is worth card time at all.

    `pairs` weights a pair by the gap between the two readings, so two studies carrying the SAME level
    form no pair and teach nothing. The reader emits five levels and the rarest findings put ~93% of
    studies on the bottom one, which strands most of their pairs. A continuous teacher separates all of
    them. Here every label score is identical, so `ranking` has nothing to say and `imitation` still does
    — if that ever stopped holding, the claimed source of the extra supervision would be gone.
    """
    batch = _taught([3.0, 2.0, 1.0, -1.0, -2.0, -3.0], [0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    assert float(RankedObjective().ranking(batch)) == 0.0, "the labels separate nothing in this batch"
    assert float(DistilledObjective().imitation(batch)) > 0.0, "the teacher separates all of it"


def test_imitation_ignores_pairs_the_teacher_does_not_separate():
    """A teacher with no opinion is not evidence of misordering, however the student ranks them.

    The boundary case of the class above: same algebra, teacher side flat instead of the label side.
    """
    order = [3.0, 2.0, 1.0, -1.0, -2.0, -3.0]

    assert float(DistilledObjective().imitation(_taught(order, [0.5] * len(order)))) == 0.0


def test_imitation_raises_when_no_teacher_arrived():
    """A distilling run whose targets never landed must FAIL, not quietly descend cross-entropy.

    The failure this repo keeps paying for is a flag recorded in a config and ignored by its consumer:
    the arm runs, the term contributes nothing, and the flat number reads as a refutation of the
    mechanism rather than as the plumbing bug it is.
    """
    with pytest.raises(ValueError, match="no teacher readings"):
        DistilledObjective().imitation(_batch([3.0, 2.0, 1.0, -1.0, -2.0, -3.0]))


def test_distilled_objective_adds_the_term_at_its_configured_strength():
    """Zero strength descends exactly what the ranking objective alone would, and more strength costs more."""
    batch = _taught([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], [0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    plain = float(RankedObjective(0.0)(Predicted(batch.logits, batch.scores, batch.weights)))

    assert float(DistilledObjective(0.0)(batch)) == pytest.approx(plain), "off is off, bit for bit"
    assert float(DistilledObjective(1.0)(batch)) > float(DistilledObjective(0.5)(batch)) > plain


def test_imitation_carries_a_gradient_to_every_study_it_misorders():
    """Every study in a contradicted batch must receive gradient, or the term trains only its extremes."""
    logits = torch.tensor([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]], requires_grad=True)
    flat = torch.full((6, 1), 0.08)
    teacher = torch.tensor([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]).reshape(-1, 1)

    DistilledObjective().imitation(Predicted(logits, flat, torch.ones_like(flat), teacher=teacher)).backward()

    assert logits.grad is not None
    assert bool((logits.grad.abs() > 0).all()), "a study with no gradient is a study the term cannot fix"
