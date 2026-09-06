import pytest
import torch

from core.findings import COMPARTMENT_PAIRS, TARGETS
from kneemri.training.grades import GRADE_LADDER, OrdinalGrades, Predicted
from kneemri.training.label_focus import FOCUS, UNFOCUSED, LabelFocus
from kneemri.training.objective import (
    CONTRAST_TERM,
    COUNT_NORMALISED,
    DISSENT_TERM,
    DISTILL_TERM,
    FEATURE_TERM,
    MARGIN_TERM,
    PAIR_POOL,
    PER_LABEL,
    POSITIVE_AT,
    RANK_TERM,
    ContrastiveObjective,
    DecorrelatedObjective,
    DistilledObjective,
    LabelStructure,
    MarginObjective,
    Objective,
    RankedObjective,
    Shaping,
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
    assert Objective(Shaping(per_label=True)).settings() == {
        RANK_TERM: 0.0,
        DISTILL_TERM: 0.0,
        FEATURE_TERM: 0.0,
        DISSENT_TERM: 0.0,
        CONTRAST_TERM: 0.0,
        MARGIN_TERM: 0.0,
        PAIR_POOL: 0,
        PER_LABEL: True,
        COUNT_NORMALISED: False,
        GRADE_LADDER: False,
        SIDE_TERM: 0.0,
        FOCUS: "",
        UNFOCUSED: 0.0,
    }


def test_ranked_objective_settings():
    """The ranked one reports the term's real strength, not the flag that asked for it."""
    assert RankedObjective(0.3, Shaping(normalise_by_count=True)).settings() == {
        RANK_TERM: 0.3,
        DISTILL_TERM: 0.0,
        FEATURE_TERM: 0.0,
        DISSENT_TERM: 0.0,
        CONTRAST_TERM: 0.0,
        MARGIN_TERM: 0.0,
        PAIR_POOL: 0,
        PER_LABEL: False,
        COUNT_NORMALISED: True,
        GRADE_LADDER: False,
        SIDE_TERM: 0.0,
        FOCUS: "",
        UNFOCUSED: 0.0,
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
    by_count = Objective(Shaping(normalise_by_count=True)).cross_entropy(Predicted(logits, scores, weights))
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
    balanced = float(Objective(Shaping(per_label=True))(Predicted(logits, scores, weights)))
    assert balanced > pooled * 4, "the rare label must dominate once every column counts the same"

    # And with the columns equally represented the two agree, so this is a REWEIGHTING and not a rescale.
    even = torch.ones(20, 2)
    pooled_even = float(Objective()(Predicted(logits, scores, even)))
    balanced_even = float(Objective(Shaping(per_label=True))(Predicted(logits, scores, even)))
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
    laddered = RankedObjective(shaping=Shaping(structure=LabelStructure(grades=OrdinalGrades())))
    plain = RankedObjective()
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
    assert LabelStructure().settings() == {GRADE_LADDER: False, SIDE_TERM: 0.0, FOCUS: "", UNFOCUSED: 0.0}
    assert LabelStructure(grades=OrdinalGrades()).settings()[GRADE_LADDER] is True
    assert LabelStructure(sides=PairedSides(0.5)).settings()[SIDE_TERM] == 0.5
    assert LabelStructure(focus=LabelFocus((TARGETS[3],))).settings()[FOCUS] == TARGETS[3]


def test_focused():
    """The structure delegates the mask, and the null policy hands the same object back.

    Identity rather than equality on the null case: every member shipped to date was fitted through
    this branch, and a mask of ones would multiply where the old runs did not.
    """
    batch = _twelve()

    assert LabelStructure().focused(batch) is batch
    assert float(LabelStructure(focus=LabelFocus((TARGETS[3],))).focused(batch).weights.sum()) == float(
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


def _contested(order: list[float], teacher: list[float]) -> Predicted:
    """`_batch`'s labelled studies, plus what an ensemble said about the same six.

    The labels stay binary and fully separated here, which is the opposite of `_taught`: the whole point
    of this term is that it pays only for pairs the LABELS order, so a batch whose labels order nothing
    would leave it with nothing to re-weight.
    """
    labelled = _batch(order)
    return Predicted(labelled.logits, labelled.scores, labelled.weights, teacher=torch.tensor(teacher).reshape(-1, 1))


def test_decorrelated_objective_settings():
    """Its account reports its own strength AND the ranking term as OFF, because it holds that at zero.

    The two keys together are the check that matters: this term IS the label ordering at a re-weighted
    mass, so a config showing both live would mean the arm descended that ordering twice.
    """
    settings = DecorrelatedObjective(0.4).settings()

    assert settings[DISSENT_TERM] == 0.4
    assert settings[RANK_TERM] == 0.0, "the flat-mass ranking term is deliberately not composed with this"
    assert settings[DISTILL_TERM] == 0.0
    assert settings[PAIR_POOL] == 0


def test_dissent():
    """Ordering the labels' way costs less than contradicting them, whatever the teacher emphasises.

    The base case, and the guard on the term's direction: the emphasis chooses WHICH pairs are charged,
    never which order is correct. A student that ranks the reports right owes almost nothing here even
    on the pairs the pool gets wrong — which is what makes the term safe to run unbounded.
    """
    objective = DecorrelatedObjective()
    teacher = [0.9, 0.5, 0.1, 0.2, 0.2, 0.2]

    agreeing = objective.dissent(_contested([3.0, 2.0, 1.0, -1.0, -2.0, -3.0], teacher))
    contradicting = objective.dissent(_contested([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], teacher))

    assert float(agreeing) < float(contradicting)
    assert float(agreeing) < 0.1, "a student already ordering the reports' way owes almost nothing"


def test_dissent_matches_plain_ranking_when_the_teacher_separates_nothing():
    """A teacher with no opinion emphasises nothing, and the term collapses onto `ranking` exactly.

    THE EQUALITY THAT SAYS THE EMPHASIS IS DOING THE WORK. Both sums carry the emphasis, so a uniform one
    cancels whatever its size — if this ever drifted, the term would be `ranking` on a rescaled axis and
    every number it produced would be a learning-rate change wearing a mechanism's name.
    """
    batch = _contested([3.0, 1.0, -1.0, 2.0, -2.0, -3.0], [0.5] * STUDIES)

    assert float(DecorrelatedObjective().dissent(batch)) == pytest.approx(
        float(RankedObjective().ranking(batch)), abs=1e-6
    )


def test_dissent_pays_more_for_the_pairs_the_teacher_gets_wrong():
    """THE MECHANISM: two students that owe `ranking` the same amount do NOT owe this term the same.

    Both batches misorder exactly one positive study by the same margin, so the flat-mass term cannot
    tell them apart. The teacher ranks study 0 confidently above the negatives and study 2 below them,
    so the student that fails where the POOL already fails is the one this term charges — and the one
    that fixes it is a member the pool is missing.
    """
    teacher = [0.9, 0.5, 0.1, 0.2, 0.2, 0.2]
    fails_where_the_pool_fails = _contested([5.0, 5.0, -5.0, 0.0, 0.0, 0.0], teacher)
    fails_where_the_pool_is_right = _contested([-5.0, 5.0, 5.0, 0.0, 0.0, 0.0], teacher)

    assert float(RankedObjective().ranking(fails_where_the_pool_fails)) == pytest.approx(
        float(RankedObjective().ranking(fails_where_the_pool_is_right)), abs=1e-6
    ), "the flat-mass term is blind to which study was misordered, which is why this term exists"
    assert float(DecorrelatedObjective().dissent(fails_where_the_pool_fails)) > float(
        DecorrelatedObjective().dissent(fails_where_the_pool_is_right)
    )


def test_dissent_raises_when_no_teacher_arrived():
    """A decorrelating run whose readings never landed must FAIL rather than descend the plain term.

    `test_imitation_raises_when_no_teacher_arrived`'s reason, and the same cost: the arm would run, the
    emphasis would never exist, and its flat number would read as a refutation of the mechanism.
    """
    with pytest.raises(ValueError, match="no teacher readings"):
        DecorrelatedObjective().dissent(_batch([3.0, 2.0, 1.0, -1.0, -2.0, -3.0]))


def test_decorrelated_objective_adds_the_term_at_its_configured_strength():
    """Zero strength descends exactly what plain cross-entropy would, and more strength costs more."""
    batch = _contested([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], [0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    plain = float(Objective()(Predicted(batch.logits, batch.scores, batch.weights)))

    assert float(DecorrelatedObjective(0.0)(batch)) == pytest.approx(plain), "off is off, bit for bit"
    assert float(DecorrelatedObjective(1.0)(batch)) > float(DecorrelatedObjective(0.5)(batch)) > plain


def test_dissent_never_asks_the_student_to_contradict_the_labels():
    """The anchor: a batch ranked exactly as the reports order it owes this term nothing, teacher or not.

    This is what separates it from a correlation penalty, and it is the property that makes the term safe
    to run unbounded — being different is only ever rewarded through being right, so a perfect student is
    a fixed point however wrong the pool is about these studies.
    """
    contrary = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]

    assert float(DecorrelatedObjective().dissent(_contested([9.0, 9.0, 9.0, -9.0, -9.0, -9.0], contrary))) == (
        pytest.approx(0.0, abs=1e-3)
    )


def _graded(levels: list[float], order: list[float]) -> Predicted:
    """One label read at the reader's five levels rather than binarised, at full confidence."""
    logits = torch.tensor(order).reshape(-1, 1)
    scores = torch.tensor(levels).reshape(-1, 1)
    return Predicted(logits, scores, torch.ones_like(scores))


def test_margin_objective_settings():
    """Its account reports the margin's WIDTH, which is also the flag that selects it, and every added
    term as off — this objective composes with none of them because it removes what they add to.
    """
    settings = MarginObjective(0.8).settings()

    assert settings[MARGIN_TERM] == 0.8
    assert settings[RANK_TERM] == 0.0
    assert settings[DISTILL_TERM] == 0.0
    assert settings[DISSENT_TERM] == 0.0


def test_auc_margin():
    """Separating the two classes by the margin costs little; ordering them backwards costs a lot."""
    objective = MarginObjective()

    separated = objective.auc_margin(_batch([3.0, 2.0, 1.0, -1.0, -2.0, -3.0]))
    reversed_order = objective.auc_margin(_batch([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]))

    assert float(separated) < float(reversed_order)
    assert float(reversed_order) > 1.0, "a fully reversed batch owes more than the whole margin squared"


def test_the_margin_cuts_where_the_scorer_cuts():
    """THE ONE WAY THIS OBJECTIVE CAN BE SILENTLY WRONG: optimising a different partition than the one
    scored. The loss binarises at `POSITIVE_AT` and the twelve holdout AUCs binarise at `LabelScores`'
    own midpoint, and nothing but this equality keeps the two the same number.
    """
    from kneemri.evaluation.scores import MIDPOINT  # noqa: PLC0415 - local to the one test that needs it

    assert POSITIVE_AT == MIDPOINT


def test_auc_margin_reads_the_cut_and_not_the_level():
    """The metric sees a positive, never HOW positive, so two batches that binarise alike must cost alike.

    Both carry the reader's real levels, and they differ on every one of them — the graded value is the
    part of the reading this term is designed to throw away, and a loss that still moved with it would
    be spending capacity on exactly the degree of freedom the objective exists to stop paying for.
    """
    objective = MarginObjective()
    order = [3.0, 2.0, 1.0, -1.0, -2.0, -3.0]

    top = objective.auc_margin(_graded([0.94, 0.82, 0.68, 0.08, 0.28, 0.08], order))
    bottom = objective.auc_margin(_graded([0.68, 0.68, 0.94, 0.28, 0.08, 0.28], order))

    assert float(top) == pytest.approx(float(bottom), abs=1e-7)


def test_auc_margin_drops_a_label_no_pair_can_be_formed_for():
    """A label all-positive in this batch forms no comparison, and must leave the mean rather than
    enter it as a zero — a zero reads as a label already separated by the full margin.
    """
    order = [3.0, 2.0, 1.0, -1.0, -2.0, -3.0]
    one = _batch(order)
    both = Predicted(
        torch.cat([one.logits, one.logits], dim=1),
        torch.cat([torch.ones_like(one.scores), one.scores], dim=1),
        torch.ones(len(order), 2),
    )

    assert float(MarginObjective().auc_margin(both)) == pytest.approx(float(MarginObjective().auc_margin(one)))


def test_auc_margin_is_zero_when_nothing_can_be_compared():
    """Every label all-positive: no term exists, and the step must be a hard zero that still carries a
    graph — returning a bare constant would break the backward pass on a batch a sampler can produce.
    """
    logits = torch.tensor([[3.0], [2.0], [1.0], [-1.0], [-2.0], [-3.0]], requires_grad=True)
    scores = torch.ones_like(logits)
    term = MarginObjective().auc_margin(Predicted(logits, scores, torch.ones_like(scores)))

    assert float(term) == 0.0
    term.backward()
    assert logits.grad is not None


def test_margin_objective_replaces_cross_entropy_rather_than_adding_to_it():
    """A batch ranked perfectly but scaled badly owes this objective almost nothing and BCE plenty.

    THE WHOLE CLAIM, in one comparison: the logits below are miscalibrated by a wide margin and ordered
    correctly with the classes fully apart. Cross-entropy charges for the calibration, this does not,
    and the metric reads the ordering only.
    """
    batch = _batch([8.0, 8.0, 8.0, -8.0, -8.0, -8.0])

    assert float(MarginObjective()(batch)) < 0.01
    assert float(Objective()(batch)) < float(MarginObjective()(batch)) + 1.0, "sanity: both are small here"
    assert float(MarginObjective()(_batch([0.2, 0.1, 0.05, -0.05, -0.1, -0.2]))) > float(MarginObjective()(batch)), (
        "a correctly ordered but barely separated batch still owes the margin"
    )


def test_auc_margin_carries_a_gradient_to_every_study_it_has_not_separated():
    """Every study must receive gradient, or the term trains only the studies at the class means."""
    logits = torch.tensor([[-1.0], [0.0], [1.0], [-1.0], [0.0], [1.0]], requires_grad=True)
    scores = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).reshape(-1, 1)

    MarginObjective().auc_margin(Predicted(logits, scores, torch.ones_like(scores))).backward()

    assert logits.grad is not None
    assert bool((logits.grad.abs() > 0).all()), "a study with no gradient is a study the term cannot fix"


def _embedded(directions: list[list[float]], positive: list[float]) -> Predicted:
    """One finding's worth of unit-ish embeddings beside the labels that say which should sit together."""
    embedding = torch.tensor(directions).unsqueeze(1)
    scores = torch.tensor(positive).reshape(-1, 1)
    logits = torch.zeros_like(scores)
    return Predicted(logits, scores, torch.ones_like(scores), embedding=embedding)


def test_contrast():
    """A representation that already groups torn with torn owes far less than one that scatters them.

    The direction is the whole claim of the term: it is a statement about the VECTOR the head read, and
    it must be blind to the logits, which are identical (zero) in both batches here.
    """
    labels = [1.0, 1.0, 0.0, 0.0]
    grouped = _embedded([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], labels)
    scattered = _embedded([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]], labels)
    objective = ContrastiveObjective()

    assert float(objective.contrast(grouped)) < float(objective.contrast(scattered))


def test_supcon():
    """Widening the key population must leave the term alone when the keys ARE the anchors.

    The equality is the contract `RememberedContrast` builds on: `contrast` is `supcon(batch, batch)`, so
    a pool that appends nothing has to reproduce it exactly, and a pool that appends a disagreeing study
    has to cost more. Anything else and the widened arm's number would carry the refactor as well as the
    mechanism.
    """
    batch = _embedded([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], [1.0, 1.0, 0.0, 0.0])
    against = _embedded([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0]], [1.0, 1.0, 0.0, 0.0, 1.0])
    objective = ContrastiveObjective()

    assert torch.equal(objective.supcon(batch, batch), objective.contrast(batch))
    assert float(objective.supcon(batch, against)) > float(objective.contrast(batch))


def test_contrast_refuses_a_batch_with_no_embedding():
    """LOUD, because the silent alternative descends plain cross-entropy under a contrastive arm's name."""
    with pytest.raises(ValueError, match="no embedding"):
        ContrastiveObjective().contrast(_batch([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]))


def test_contrastive_objective_settings():
    """It reports the term's real strength, so a queued arm can be checked for it before the card."""
    assert ContrastiveObjective(0.5).settings()[CONTRAST_TERM] == 0.5
    assert Objective().settings()[CONTRAST_TERM] == 0.0, "the plain one reports it OFF, never omits it"


def test_contrastive_objective_carries_the_ranking_term():
    """The house recipe passes --rank-weight, so dropping it would make every contrastive arm two changes.

    Both halves are asserted: the settings must SAY the ranking term is on, and the scalar must actually
    differ from the same objective with the term at zero — a reported weight that changes no loss is the
    silent no-op this file's guards exist against.
    """
    composed = ContrastiveObjective(1.0, rank_weight=1.0)
    batch = _embedded([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], [1.0, 1.0, 0.0, 0.0])

    assert composed.settings()[RANK_TERM] == 1.0
    assert ContrastiveObjective(1.0).settings()[RANK_TERM] == 0.0
    assert float(composed(batch)) != float(ContrastiveObjective(1.0)(batch))


def test_attached():
    """Every term but one owns NO weights, and the base hook has to say so rather than be optional.

    A hook that objectives could omit would put the burden on the optimiser's builder to know which
    class holds tensors; answering an empty list from the base makes the composition total, and the one
    subclass that does own a projection is then the only place the question is answered differently.
    """
    for objective in (Objective(), RankedObjective(0.3), DistilledObjective(0.3), ContrastiveObjective(0.5)):
        assert objective.attached(torch.device("cpu")) == []
