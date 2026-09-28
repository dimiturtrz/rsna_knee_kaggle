import torch

from core.findings import COMPARTMENT_PAIRS, TARGETS
from kneemri.training.grades import GRADE_LADDER, OrdinalGrades, Predicted
from kneemri.training.label_focus import FOCUS, UNFOCUSED, LabelFocus
from kneemri.training.objective import COUNT_NORMALISED, PER_LABEL, LabelStructure, Objective, Shaping
from kneemri.training.side_ranking import SIDE_TERM, PairedSides
from kneemri.training.terms import (
    CONTRAST_TERM,
    DISSENT_TERM,
    DISTILL_TERM,
    EARLY_TERM,
    FEATURE_TERM,
    MARGIN_TERM,
    PAIR_POOL,
    RANK_TERM,
    Contrast,
    CrossEntropy,
    Dissent,
    Distillation,
    Margin,
    Ranking,
)


def _batch(order: list[float]) -> Predicted:
    """One label's worth of logits laid out in `order`, with binary targets and full confidence."""
    logits = torch.tensor(order).reshape(-1, 1)
    scores = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
    return Predicted(logits, scores, torch.ones_like(scores))


def _embedded(directions: list[list[float]], positive: list[float]) -> Predicted:
    """One finding's worth of embeddings beside the labels that say which should sit together."""
    embedding = torch.tensor(directions).unsqueeze(1)
    scores = torch.tensor(positive).reshape(-1, 1)
    return Predicted(torch.zeros_like(scores), scores, torch.ones_like(scores), embedding=embedding)


def test_objective_settings():
    """The plain objective's own account of its loss switches, which a run checks before it starts.

    It reports the ranking term OFF rather than omitting the key: the question is whether a ranking arm
    got a ranking objective at all, and a missing key cannot answer it.
    """
    assert Objective(shaping=Shaping(per_label=True)).settings() == {
        RANK_TERM: 0.0,
        EARLY_TERM: 0.0,
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


def test_objective_reports_each_term_at_its_real_strength():
    """A composed objective reports each term's real strength, not the flag that asked for it."""
    assert Objective(terms=[Ranking(0.3)], shaping=Shaping(normalise_by_count=True)).settings() == {
        RANK_TERM: 0.3,
        EARLY_TERM: 0.0,
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
    entropy = CrossEntropy()

    wrong = entropy(Predicted(torch.tensor([[-4.0, 4.0], [4.0, -4.0]]), scores, weights))
    right = entropy(Predicted(torch.tensor([[4.0, -4.0], [-4.0, 4.0]]), scores, weights))
    assert right < wrong
    assert float(right) > 0.0


def test_cross_entropy_normalisation_modes_differ():
    """Dividing by total weight is not the same step as dividing by element count.

    With most weights below one the by-weight form gives the larger gradient for the same batch, which
    is why the two cannot be swapped without re-measuring the learning rate alongside.
    """
    batch = Predicted(torch.zeros(4, 2), torch.ones(4, 2), torch.full((4, 2), 0.25))

    by_weight = CrossEntropy()(batch)
    by_count = CrossEntropy(normalise_by_count=True)(batch)
    assert not torch.isclose(by_weight, by_count)
    assert by_weight > by_count


def test_cross_entropy_per_label_gives_every_label_the_same_say():
    """Pooled, a label's share of the gradient tracks how OFTEN it is mentioned; the metric does not.

    Effusion is positive in 60% of studies and Fracture in 7%, so the pooled objective spends about ten
    times more on one than the other while the score averages the twelve equally. Normalising per column
    removes that, which is the only sense in which the loss can be said to match the metric.
    """
    common, rare = 0, 1
    logits = torch.zeros(20, 2)
    logits[:, common] = 3.0
    logits[0, rare] = -3.0
    scores = torch.ones(20, 2)
    weights = torch.zeros(20, 2)
    weights[:, common] = 1.0
    weights[0, rare] = 1.0
    per_label = Objective(shaping=Shaping(per_label=True))

    pooled = float(Objective()(Predicted(logits, scores, weights)))
    balanced = float(per_label(Predicted(logits, scores, weights)))
    assert balanced > pooled * 4, "the rare label must dominate once every column counts the same"

    even = torch.ones(20, 2)
    pooled_even = float(Objective()(Predicted(logits, scores, even)))
    balanced_even = float(per_label(Predicted(logits, scores, even)))
    assert abs(pooled_even - balanced_even) < 1e-6, "equal representation, equal answer — a reweighting, not a rescale"


def test_ranking_at_zero_strength_is_plain_cross_entropy():
    """Turning the term off must leave the measured default bit-for-bit unchanged."""
    batch = _batch([-6.0, -5.0, -4.0, 4.0, 5.0, 6.0])
    assert torch.equal(Objective(terms=[Ranking(0.0)])(batch), Objective()(batch))


def test_ordinal_grades_leave_the_terms_alone():
    """The ladder is a BASE policy, and every term still reads the twelve latents.

    THIS IS THE SEPARATION THE DESIGN RESTS ON: applied to the terms too, a ranking arm would be ordering
    forty-eight threshold crossings against each other, which is a different experiment wearing this
    one's name.
    """
    batch = _batch([1.0, 0.5, 0.0, -0.5, -1.0, -1.5])
    ladder = Shaping(structure=LabelStructure(grades=OrdinalGrades()))
    laddered = Objective(terms=[Ranking(0.5)], shaping=ladder)
    plain = Objective(terms=[Ranking(0.5)])

    assert torch.isclose(
        laddered(batch) - Objective(shaping=ladder)(batch), plain(batch) - Objective()(batch), atol=1e-6
    ), "the ranking term's contribution is the same with or without the ladder"
    assert not torch.isclose(Objective(shaping=ladder)(batch), Objective()(batch))


def test_margin_base_replaces_cross_entropy_rather_than_adding_to_it():
    """A batch ranked perfectly but scaled badly owes the margin base almost nothing.

    THE WHOLE CLAIM, in one comparison: the logits below are miscalibrated by a wide margin and ordered
    correctly with the classes fully apart. Cross-entropy charges for the calibration, this does not.
    """
    batch = _batch([8.0, 8.0, 8.0, -8.0, -8.0, -8.0])
    margin = Objective(base=Margin())

    assert torch.equal(margin(batch), Margin()(batch)), "the base is summed unscaled, and nothing else is added"
    assert float(margin(batch)) < 0.01
    assert float(margin(_batch([0.2, 0.1, 0.05, -0.05, -0.1, -0.2]))) > float(margin(batch))
    assert Objective(base=Margin(0.8)).settings()[MARGIN_TERM] == 0.8


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


def test_spread():
    """The null grade policy hands the batch back untouched; the ladder restates it."""
    batch = _twelve()

    assert LabelStructure().spread(batch).logits.shape == batch.logits.shape
    assert LabelStructure(grades=OrdinalGrades()).spread(batch).logits.shape[-1] > batch.logits.shape[-1]


def test_side_term():
    """A hard zero under the null policy, so an objective built without it descends the old scalar."""
    batch = _twelve()

    assert float(LabelStructure().side_term(batch)) == 0.0
    assert float(LabelStructure(sides=PairedSides(1.0)).side_term(batch)) > 0.0


def test_objective_attached():
    """An objective owns exactly what its base and terms own — here nothing."""
    cpu = torch.device("cpu")
    for objective in (
        Objective(),
        Objective(terms=[Ranking(0.3)]),
        Objective(terms=[Ranking(0.3), Distillation(0.3)]),
        Objective(terms=[Dissent(0.3)]),
        Objective(terms=[Contrast(0.5)]),
        Objective(base=Margin()),
    ):
        assert objective.attached(cpu) == []
