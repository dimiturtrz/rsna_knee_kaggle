from collections.abc import Callable

import pytest
import torch

from core.findings import TARGETS
from kneemri.evaluation.scores import MIDPOINT
from kneemri.training.grades import OrdinalGrades, Predicted
from kneemri.training.label_focus import LabelFocus
from kneemri.training.objective import LabelStructure, Objective, Shaping
from kneemri.training.side_ranking import PairedSides
from kneemri.training.terms import (
    CONTRAST_TERM,
    DISSENT_TERM,
    DISTILL_TERM,
    EARLY_TERM,
    POSITIVE_AT,
    RANK_TERM,
    Contrast,
    CrossEntropy,
    Dissent,
    Distillation,
    EarlyLearning,
    Margin,
    PairMass,
    Ranking,
    Term,
    Widening,
)

SEED = 2026
STUDIES = 6
LABELS = len(TARGETS)


def _batch(order: list[float]) -> Predicted:
    """One label's worth of logits laid out in `order`, with binary targets and full confidence."""
    logits = torch.tensor(order).reshape(-1, 1)
    scores = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
    return Predicted(logits, scores, torch.ones_like(scores))


def _taught(order: list[float], teacher: list[float]) -> Predicted:
    """One label's logits in `order`, judged against a teacher that ranks them by `teacher`.

    The label scores are held IDENTICAL across every study on purpose: a pair the reports do not separate
    contributes nothing to `Ranking`, so any signal these batches carry can only have come from the teacher.
    """
    logits = torch.tensor(order).reshape(-1, 1)
    flat = torch.full((len(order), 1), 0.08)
    return Predicted(logits, flat, torch.ones_like(flat), teacher=torch.tensor(teacher).reshape(-1, 1))


def _contested(order: list[float], teacher: list[float]) -> Predicted:
    """`_batch`'s labelled studies, plus what an ensemble said about the same six."""
    labelled = _batch(order)
    return Predicted(labelled.logits, labelled.scores, labelled.weights, teacher=torch.tensor(teacher).reshape(-1, 1))


def _graded(levels: list[float], order: list[float]) -> Predicted:
    """One label read at the reader's five levels rather than binarised, at full confidence."""
    logits = torch.tensor(order).reshape(-1, 1)
    scores = torch.tensor(levels).reshape(-1, 1)
    return Predicted(logits, scores, torch.ones_like(scores))


def _embedded(directions: list[list[float]], positive: list[float]) -> Predicted:
    """One finding's worth of unit-ish embeddings beside the labels that say which should sit together."""
    embedding = torch.tensor(directions).unsqueeze(1)
    scores = torch.tensor(positive).reshape(-1, 1)
    return Predicted(torch.zeros_like(scores), scores, torch.ones_like(scores), embedding=embedding)


def _everything() -> Predicted:
    """A seeded batch carrying every field any term reads, so one batch can price each term."""
    torch.manual_seed(SEED)
    return Predicted(
        logits=torch.randn(STUDIES, LABELS),
        scores=torch.rand(STUDIES, LABELS),
        weights=torch.rand(STUDIES, LABELS),
        teacher=torch.rand(STUDIES, LABELS),
        embedding=torch.randn(STUDIES, LABELS, 4),
    )


ADDED: list[Callable[[], Term]] = [
    lambda: Ranking(0.3),
    lambda: Distillation(0.7),
    lambda: Dissent(0.4),
    lambda: Contrast(0.5),
]
SHAPINGS: list[Shaping] = [
    Shaping(),
    Shaping(per_label=True),
    Shaping(normalise_by_count=True, structure=LabelStructure(grades=OrdinalGrades(), sides=PairedSides(0.5))),
    Shaping(structure=LabelStructure(focus=LabelFocus(("Effusion",), unfocused=0.25))),
]


@pytest.mark.parametrize("shaping", SHAPINGS)
@pytest.mark.parametrize("make", ADDED)
def test_objective_is_cross_entropy_plus_each_term_at_its_strength(make: Callable[[], Term], shaping: Shaping) -> None:
    """THE EQUIVALENCE ORACLE, kept: an added term is exactly `w * term(focused)` on top of the base.

    Every term reads the FOCUSED batch and never the spread one, and the base reads the spread one — so
    this equality, on every shaping, is what says the refactor composes rather than re-derives the loss.
    """
    batch = _everything()
    term = make()
    structure = shaping.structure
    focused = structure.focused(batch)
    expected = (
        CrossEntropy(per_label=shaping.per_label, normalise_by_count=shaping.normalise_by_count)(
            structure.spread(focused)
        )
        + structure.side_term(focused)
        + term.weight * term(focused)
    )

    assert torch.equal(Objective(terms=[term], shaping=shaping)(batch), expected)


@pytest.mark.parametrize("shaping", SHAPINGS)
def test_margin_base_is_the_margin_on_the_spread_batch(shaping: Shaping) -> None:
    """The base swaps cross-entropy out; the side term still rides on top, and nothing scales the base."""
    batch = _everything()
    structure = shaping.structure
    focused = structure.focused(batch)
    expected = Margin(0.8)(structure.spread(focused)) + structure.side_term(focused)

    assert torch.equal(Objective(base=Margin(0.8), shaping=shaping)(batch), expected)


def test_ratio():
    """The share of the pair mass misordered, with a floored denominator so an empty batch is zero."""
    assert float(PairMass(torch.tensor(1.0), torch.tensor(4.0)).ratio()) == 0.25
    assert float(PairMass(torch.tensor(0.0), torch.tensor(0.0)).ratio()) == 0.0


def test_between():
    """The (numerator, denominator) split must be the ratio `Ranking` returns, and pooling ADDS both."""
    batch = _batch([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    mass = PairMass.between(batch, batch)

    assert float(mass.total) > 0.0
    assert torch.allclose(mass.misordered / mass.total, Ranking()(batch))
    doubled = mass + mass
    assert torch.equal(doubled.total, mass.total * 2)
    assert torch.allclose(doubled.ratio(), mass.ratio())


def test_ranking():
    """A batch ordered the right way costs near nothing; the same batch reversed costs a lot."""
    assert float(Ranking()(_batch([6.0, 5.0, 4.0, -4.0, -5.0, -6.0]))) < 0.01
    assert float(Ranking()(_batch([-6.0, -5.0, -4.0, 4.0, 5.0, 6.0]))) > 1.0


def test_ranking_is_blind_to_a_shift_that_leaves_the_order_alone():
    """The term reads ORDER, so adding a constant to every logit changes nothing; cross-entropy does move."""
    batch = _batch([6.0, 5.0, 4.0, -4.0, -5.0, -6.0])
    lifted = Predicted(batch.logits + 3.0, batch.scores, batch.weights)
    entropy = CrossEntropy()

    assert torch.isclose(Ranking()(batch), Ranking()(lifted), atol=1e-6)
    assert not torch.isclose(entropy(batch), entropy(lifted))


def test_ranking_ignores_pairs_the_reports_do_not_separate():
    """Two studies read the same way form no pair, so a batch with one reading costs nothing."""
    same = torch.full((3, 1), 0.7)
    assert float(Ranking()(Predicted(torch.tensor([[3.0], [-3.0], [0.5]]), same, torch.ones_like(same)))) == 0.0


def test_ranking_discounts_a_pair_resting_on_a_guess():
    """A pair is weighted by BOTH readings' confidence, so an unconfident study barely pulls."""
    batch = _batch([-6.0, -5.0, -4.0, 4.0, 5.0, 6.0])
    confident = Ranking()(batch)
    unsure = Ranking()(Predicted(batch.logits, batch.scores, batch.weights * 0.1))
    assert torch.isclose(unsure, confident), "the term is a weighted MEAN, so a uniform discount cancels"

    partial = batch.weights.clone()
    partial[0] = 0.0
    assert float(Ranking()(Predicted(batch.logits, batch.scores, partial))) < float(confident)


def test_ranking_carries_a_gradient_to_every_study_in_a_misordered_batch():
    """A loss that cannot reach the logits is not training anything — the failure a value test misses."""
    torch.manual_seed(SEED)
    logits = torch.randn(STUDIES, LABELS, requires_grad=True)
    scores = torch.randint(0, 2, (STUDIES, LABELS)).float()
    Ranking()(Predicted(logits, scores, torch.ones_like(scores))).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0


def test_ranking_across():
    """A pair no single batch contains must still reach the loss once the earlier batch is remembered.

    Both batches are internally ordered perfectly, so the within-batch term is near zero for each. Across
    them the ordering is inverted — the corpus-level misordering an AUC punishes and a per-batch loss
    cannot see. `softplus` never reaches zero, so "internally perfect" means a gap of ~14.
    """
    early = _batch([9.0, 8.0, 7.0, -7.0, -8.0, -9.0])
    late = _batch([-20.0, -21.0, -22.0, -36.0, -37.0, -38.0])

    assert float(Ranking()(late)) < 1e-3, "the late batch is internally perfect"
    assert float(Ranking().across(late, early)) > 0.1, "and yet its positives sit below the earlier negatives"


def test_margin_across():
    """Remembering the batch's own twin leaves the class means, and so the margin, unchanged."""
    batch = _batch([0.4, -1.0, 2.0, 0.5, 3.0, -2.0])
    assert torch.allclose(Margin().across(batch, batch), Margin()(batch), atol=1e-6)


def test_contrast_across():
    """A remembered positive the live one disagrees with must be charged for."""
    live = _embedded([[-1.0, 0.0], [0.0, 1.0]], [1.0, 0.0])
    remembered = _embedded([[1.0, 0.0]], [1.0])
    assert float(Contrast().across(live, remembered)) > float(Contrast()(live)), (
        "a remembered positive the live one disagrees with must be charged for"
    )


def test_distillation():
    """Agreeing with the teacher's order costs less than contradicting it, which is the whole term."""
    teacher = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    agreeing = Distillation()(_taught([3.0, 2.0, 1.0, -1.0, -2.0, -3.0], teacher))
    contradicting = Distillation()(_taught([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], teacher))

    assert float(agreeing) < float(contradicting)
    assert float(agreeing) < 0.1, "a student that already ranks the teacher's way owes almost nothing"


def test_distillation_orders_pairs_the_labels_cannot():
    """THE MECHANISM: every label score is identical, so `Ranking` has nothing to say and this still does."""
    batch = _taught([3.0, 2.0, 1.0, -1.0, -2.0, -3.0], [0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    assert float(Ranking()(batch)) == 0.0, "the labels separate nothing in this batch"
    assert float(Distillation()(batch)) > 0.0, "the teacher separates all of it"


def test_distillation_ignores_pairs_the_teacher_does_not_separate():
    """A teacher with no opinion is not evidence of misordering, however the student ranks them."""
    order = [3.0, 2.0, 1.0, -1.0, -2.0, -3.0]
    assert float(Distillation()(_taught(order, [0.5] * len(order)))) == 0.0


@pytest.mark.parametrize("make", [Distillation, Dissent])
def test_teacher_terms_raise_when_no_teacher_arrived(make: Callable[[], Term]) -> None:
    """A run whose teacher never landed must FAIL, not quietly descend the plain recipe under its name."""
    with pytest.raises(ValueError, match="no teacher readings"):
        make()(_batch([3.0, 2.0, 1.0, -1.0, -2.0, -3.0]))


def test_distillation_carries_a_gradient_to_every_study_it_misorders():
    """Every study in a contradicted batch must receive gradient, or the term trains only its extremes."""
    logits = torch.tensor([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]], requires_grad=True)
    flat = torch.full((6, 1), 0.08)
    teacher = torch.tensor([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]).reshape(-1, 1)
    Distillation()(Predicted(logits, flat, torch.ones_like(flat), teacher=teacher)).backward()

    assert logits.grad is not None
    assert bool((logits.grad.abs() > 0).all()), "a study with no gradient is a study the term cannot fix"


def _remembered(order: list[float], past: float) -> Predicted:
    """One label's logits in `order`, each study remembered at probability `past` on every earlier visit."""
    labelled = _batch(order)
    early = torch.stack([torch.full_like(labelled.logits, past), torch.full_like(labelled.logits, 1.0 - past)], dim=-1)
    return Predicted(labelled.logits, labelled.scores, labelled.weights, early=early)


def test_early_learning():
    """Agreeing with the past costs less than contradicting it — the term is a pull toward the history."""
    confident = [3.0] * STUDIES
    agreeing = EarlyLearning(1.0)(_remembered(confident, 0.99))
    contradicting = EarlyLearning(1.0)(_remembered(confident, 0.01))

    assert float(agreeing) < float(contradicting)


def test_early_learning_owes_nothing_on_a_first_visit():
    """An empty history is exactly zero agreement, so the term is silent until a study has been seen."""
    labelled = _batch([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
    empty = torch.zeros(*labelled.logits.shape, 2)
    first = Predicted(labelled.logits, labelled.scores, labelled.weights, early=empty)

    assert float(EarlyLearning(1.0)(first)) == 0.0


def test_early_learning_raises_when_no_history_arrived():
    """A run whose history never attached must FAIL, not descend the plain recipe under its name."""
    with pytest.raises(ValueError, match="early-learning targets"):
        EarlyLearning(1.0)(_batch([3.0, 2.0, 1.0, -1.0, -2.0, -3.0]))


def test_early_learning_carries_a_gradient_toward_the_history():
    """The gradient moves each logit TOWARD what the study was read as before."""
    labelled = _remembered([3.0, 2.0, 1.0, -1.0, -2.0, -3.0], 0.99)
    logits = labelled.logits.clone().requires_grad_(True)
    EarlyLearning(1.0)(Predicted(logits, labelled.scores, labelled.weights, early=labelled.early)).backward()

    assert logits.grad is not None
    assert (logits.grad < 0).all(), "descending the term raises every logit toward the remembered 0.99"


def test_dissent():
    """Ordering the labels' way costs less than contradicting them, whatever the teacher emphasises."""
    teacher = [0.9, 0.5, 0.1, 0.2, 0.2, 0.2]
    agreeing = Dissent()(_contested([3.0, 2.0, 1.0, -1.0, -2.0, -3.0], teacher))
    contradicting = Dissent()(_contested([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], teacher))

    assert float(agreeing) < float(contradicting)
    assert float(agreeing) < 0.1, "a student already ordering the reports' way owes almost nothing"


def test_dissent_matches_plain_ranking_when_the_teacher_separates_nothing():
    """A uniform emphasis cancels in both sums, so the term collapses onto `Ranking` exactly."""
    batch = _contested([3.0, 1.0, -1.0, 2.0, -2.0, -3.0], [0.5] * STUDIES)
    assert float(Dissent()(batch)) == pytest.approx(float(Ranking()(batch)), abs=1e-6)


def test_dissent_pays_more_for_the_pairs_the_teacher_gets_wrong():
    """Two students owing `Ranking` the same amount do NOT owe this term the same."""
    teacher = [0.9, 0.5, 0.1, 0.2, 0.2, 0.2]
    fails_where_the_pool_fails = _contested([5.0, 5.0, -5.0, 0.0, 0.0, 0.0], teacher)
    fails_where_the_pool_is_right = _contested([-5.0, 5.0, 5.0, 0.0, 0.0, 0.0], teacher)

    assert float(Ranking()(fails_where_the_pool_fails)) == pytest.approx(
        float(Ranking()(fails_where_the_pool_is_right)), abs=1e-6
    ), "the flat-mass term is blind to which study was misordered, which is why this term exists"
    assert float(Dissent()(fails_where_the_pool_fails)) > float(Dissent()(fails_where_the_pool_is_right))


def test_dissent_never_asks_the_student_to_contradict_the_labels():
    """A batch ranked exactly as the reports order it owes this term nothing, teacher or not."""
    contrary = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
    owed = Dissent()(_contested([9.0, 9.0, 9.0, -9.0, -9.0, -9.0], contrary))
    assert float(owed) == pytest.approx(0.0, abs=1e-3)


def test_margin():
    """Separating the two classes by the margin costs little; ordering them backwards costs a lot."""
    separated = Margin()(_batch([3.0, 2.0, 1.0, -1.0, -2.0, -3.0]))
    reversed_order = Margin()(_batch([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]))

    assert float(separated) < float(reversed_order)
    assert float(reversed_order) > 1.0, "a fully reversed batch owes more than the whole margin squared"


def test_the_margin_cuts_where_the_scorer_cuts():
    """The loss binarises at `POSITIVE_AT` and the holdout AUCs at `MIDPOINT`; only this keeps them one number."""
    assert POSITIVE_AT == MIDPOINT


def test_margin_reads_the_cut_and_not_the_level():
    """The metric sees a positive, never HOW positive, so two batches that binarise alike must cost alike."""
    order = [3.0, 2.0, 1.0, -1.0, -2.0, -3.0]
    top = Margin()(_graded([0.94, 0.82, 0.68, 0.08, 0.28, 0.08], order))
    bottom = Margin()(_graded([0.68, 0.68, 0.94, 0.28, 0.08, 0.28], order))

    assert float(top) == pytest.approx(float(bottom), abs=1e-7)


def test_margin_drops_a_label_no_pair_can_be_formed_for():
    """A label all-positive in this batch leaves the mean rather than entering it as a zero."""
    one = _batch([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
    both = Predicted(
        torch.cat([one.logits, one.logits], dim=1),
        torch.cat([torch.ones_like(one.scores), one.scores], dim=1),
        torch.ones(STUDIES, 2),
    )
    assert float(Margin()(both)) == pytest.approx(float(Margin()(one)))


def test_margin_is_zero_when_nothing_can_be_compared():
    """A hard zero that still carries a graph — a bare constant would break the backward pass."""
    logits = torch.tensor([[3.0], [2.0], [1.0], [-1.0], [-2.0], [-3.0]], requires_grad=True)
    scores = torch.ones_like(logits)
    term = Margin()(Predicted(logits, scores, torch.ones_like(scores)))

    assert float(term) == 0.0
    term.backward()
    assert logits.grad is not None


def test_margin_carries_a_gradient_to_every_study_it_has_not_separated():
    """Every study must receive gradient, or the term trains only the studies at the class means."""
    logits = torch.tensor([[-1.0], [0.0], [1.0], [-1.0], [0.0], [1.0]], requires_grad=True)
    scores = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
    Margin()(Predicted(logits, scores, torch.ones_like(scores))).backward()

    assert logits.grad is not None
    assert bool((logits.grad.abs() > 0).all()), "a study with no gradient is a study the term cannot fix"


def test_contrast():
    """A representation that groups torn with torn owes far less than one that scatters them, logits equal."""
    labels = [1.0, 1.0, 0.0, 0.0]
    grouped = _embedded([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], labels)
    scattered = _embedded([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]], labels)

    assert float(Contrast()(grouped)) < float(Contrast()(scattered))


def test_supcon():
    """`Contrast` IS `supcon(batch, batch)`, and a key population with a disagreeing study costs more."""
    batch = _embedded([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], [1.0, 1.0, 0.0, 0.0])
    against = _embedded([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0]], [1.0, 1.0, 0.0, 0.0, 1.0])

    assert torch.equal(Contrast().supcon(batch, batch), Contrast()(batch))
    assert float(Contrast().supcon(batch, against)) > float(Contrast()(batch))


def test_contrast_refuses_a_batch_with_no_embedding():
    """LOUD, because the silent alternative descends plain cross-entropy under a contrastive arm's name."""
    with pytest.raises(ValueError, match="no embedding"):
        Contrast()(_batch([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]))


def test_contrast_beside_ranking_moves_the_loss():
    """The house recipe passes --rank-weight, so the carried ranking term must change the scalar."""
    batch = _embedded([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], [1.0, 1.0, 0.0, 0.0])
    composed = Objective(terms=[Ranking(1.0), Contrast(1.0)])

    assert float(composed(batch)) != float(Objective(terms=[Contrast(1.0)])(batch))


def test_term_settings():
    """Each term reports its real strength under its own key, not the flag that asked for it."""
    assert Ranking(0.3).settings() == {RANK_TERM: 0.3}
    assert Distillation(0.4).settings() == {DISTILL_TERM: 0.4}
    assert Dissent(0.5).settings() == {DISSENT_TERM: 0.5}
    assert Contrast(0.6).settings() == {CONTRAST_TERM: 0.6}
    assert EarlyLearning(3.0).settings() == {EARLY_TERM: 3.0}


def test_cross_entropy_settings():
    """Cross-entropy adds no key of its own: the base account already carries both pooling switches."""
    assert CrossEntropy().settings() == {}


def test_graded():
    """A term read as a base is read on the SPREAD batch, which is where the grade ladder lives."""
    batch = _batch([1.0, 0.5, 0.0, -0.5, -1.0, -1.5])
    ladder = OrdinalGrades()
    entropy = CrossEntropy()

    assert torch.equal(entropy.graded(batch, ladder.spread), entropy(ladder.spread(batch)))
    assert torch.equal(entropy.graded(batch, LabelStructure().spread), entropy(batch))


def test_arrived():
    """A missing input is a REFUSAL naming the term, never a zero contribution."""
    present = torch.ones(2, 1)

    assert Distillation().arrived(present, "teacher readings") is present
    with pytest.raises(ValueError, match="distill_weight was handed a batch with no teacher readings"):
        Distillation().arrived(None, "teacher readings")


def test_attached():
    """Every term but one owns NO weights, and the base hook has to say so rather than be optional.

    A hook that terms could omit would put the burden on the optimiser's builder to know which class holds
    tensors; answering an empty list from the base makes the composition total.
    """
    cpu = torch.device("cpu")
    for term in (Ranking(0.3), Distillation(0.3), Dissent(0.3), Contrast(0.5), Margin()):
        assert term.attached(cpu) == []


def test_readout():
    """The live batch comes first and the pool follows, and no teacher is carried across."""
    live = _batch([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
    pooled = _batch([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    taught = Predicted(live.logits, live.scores, live.weights, teacher=live.scores)
    widened = Widening(taught, pooled).readout()

    assert widened.logits.shape[0] == live.logits.shape[0] + pooled.logits.shape[0]
    assert torch.equal(widened.logits[: live.logits.shape[0]], live.logits)
    assert torch.equal(widened.scores[live.scores.shape[0] :], pooled.scores)
    assert widened.teacher is None, "no pooled term reads a teacher, so carrying half of one would be a lie"


def test_embedding():
    """Both contexts as one, and None the moment either side lacks one — never the live half alone."""
    batch = _embedded([[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0])
    remembered = _embedded([[0.5, 0.5]], [1.0])
    bare = Predicted(batch.logits, batch.scores, batch.weights)

    joined = Widening(batch, remembered).embedding()
    assert joined is not None
    assert joined.shape[0] == 3
    assert torch.equal(joined[:2], batch.embedding)
    assert Widening(batch, bare).embedding() is None, "a pool with no context must raise downstream, not narrow"
    assert Widening(bare, remembered).embedding() is None
