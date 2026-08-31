"""Cross-step pairing: the equality oracle first, then the pair no single batch contains."""

import torch

from kneemri.training.grades import GRADE_LADDER, Predicted
from kneemri.training.label_focus import FOCUS
from kneemri.training.objective import (
    CONTRAST_TERM,
    COUNT_NORMALISED,
    DISSENT_TERM,
    DISTILL_TERM,
    FEATURE_TERM,
    MARGIN_TERM,
    PAIR_POOL,
    PER_LABEL,
    RANK_TERM,
    RankedObjective,
)
from kneemri.training.remembered_ranking import RememberedRanking
from kneemri.training.side_ranking import SIDE_TERM


def _batch(order: list[float]) -> Predicted:
    """One label's worth of logits laid out in `order`, with binary targets and full confidence."""
    logits = torch.tensor(order).reshape(-1, 1)
    scores = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
    return Predicted(logits, scores, torch.ones_like(scores))


def test_settings():
    """The pool is read off the memory, so a run's `--pair-memory` is checked against what would drop."""
    assert RememberedRanking(0.3, capacity=4).settings() == {
        RANK_TERM: 0.3,
        DISTILL_TERM: 0.0,
        FEATURE_TERM: 0.0,
        DISSENT_TERM: 0.0,
        CONTRAST_TERM: 0.0,
        MARGIN_TERM: 0.0,
        PAIR_POOL: 4,
        PER_LABEL: False,
        COUNT_NORMALISED: False,
        GRADE_LADDER: False,
        SIDE_TERM: 0.0,
        FOCUS: "",
    }


def test_remembered_ranking_ranking():
    """The empty memory is the equality oracle: with nothing remembered this MUST be the old objective.

    Any difference here is the new code path changing the loss for a run that has not yet remembered
    anything, which would make every comparison against the measured baseline a comparison of two things.
    """
    batch = _batch([0.4, -1.0, 2.0, 0.5, 3.0, -2.0])
    assert torch.equal(RememberedRanking().ranking(batch), RankedObjective().ranking(batch))


def test_remembered_ranking_pools_cross_step_pairs_into_one_population():
    """Remembering the same batch again must give the same ratio — pairs are added, not averaged.

    A batch remembered and then re-presented forms the identical pair population three times over
    (within, live-above-remembered, remembered-above-live), so summing numerators and denominators
    before dividing has to leave the ratio untouched. Averaging the ratios would too; what this pins is
    that BOTH directions are formed, since dropping either one leaves an asymmetric population whose
    ratio moves.
    """
    order = [0.4, -1.0, 2.0, 0.5, 3.0, -2.0]
    objective = RememberedRanking()
    first = objective.ranking(_batch(order))
    repeated = objective.ranking(_batch(order))

    assert torch.allclose(repeated, first, atol=1e-6)


def test_remembered_ranking_penalises_a_study_misordered_only_against_an_earlier_batch():
    """The whole point: a pair no single batch contains must still reach the loss.

    Both batches are internally ordered perfectly, so the within-batch term is zero for each and the
    old objective sees nothing wrong. Across them the ordering is inverted — every positive in the
    second batch sits below every negative in the first — which is exactly the corpus-level misordering
    an AUC punishes and a per-batch loss cannot see.
    """
    # Separated by far more than `softplus` can still charge for: the surrogate is smooth and never
    # reaches zero, so "internally perfect" means a gap of ~14, not a gap of 1.
    early = _batch([9.0, 8.0, 7.0, -7.0, -8.0, -9.0])
    late = _batch([-20.0, -21.0, -22.0, -36.0, -37.0, -38.0])

    objective = RememberedRanking()
    assert float(objective.ranking(early)) < 1e-3, "an internally perfect batch has nothing to pay"
    within_batch_only = float(RankedObjective().ranking(late))
    across_steps = float(objective.ranking(late))

    assert within_batch_only < 1e-3, "the late batch is internally perfect too"
    assert across_steps > 0.1, "and yet its positives are ranked below the earlier batch's negatives"
