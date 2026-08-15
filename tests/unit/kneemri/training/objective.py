import torch

from kneemri.training.objective import Objective, Predicted, RankedObjective

SEED = 2026
STUDIES = 6
LABELS = 3


def _batch(order: list[float]) -> Predicted:
    """One label's worth of logits laid out in `order`, with binary targets and full confidence."""
    logits = torch.tensor(order).reshape(-1, 1)
    scores = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
    return Predicted(logits, scores, torch.ones_like(scores))


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


def test_ranking():
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
