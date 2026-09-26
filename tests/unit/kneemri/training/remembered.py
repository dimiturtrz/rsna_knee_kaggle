from collections.abc import Callable

import pytest
import torch

from kneemri.training.grades import OrdinalGrades, Predicted
from kneemri.training.objective import LabelStructure
from kneemri.training.remembered import Remembered
from kneemri.training.terms import PAIR_POOL, Contrast, Margin, Pooled, Ranking

SEED = 2026
STUDIES = 6
LABELS = 3

POOLED: list[Callable[[], Pooled]] = [lambda: Ranking(0.3), lambda: Margin(0.8), lambda: Contrast(0.5)]


def _step(seed: int) -> Predicted:
    """A seeded batch carrying every field a pooled term reads."""
    torch.manual_seed(seed)
    return Predicted(
        logits=torch.randn(STUDIES, LABELS),
        scores=torch.rand(STUDIES, LABELS),
        weights=torch.rand(STUDIES, LABELS),
        embedding=torch.randn(STUDIES, LABELS, 4),
    )


def _batch(order: list[float]) -> Predicted:
    """One label's worth of logits laid out in `order`, with binary targets and full confidence."""
    logits = torch.tensor(order).reshape(-1, 1)
    scores = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
    return Predicted(logits, scores, torch.ones_like(scores))


@pytest.mark.parametrize("make", POOLED)
def test_remembered_is_the_bare_term_until_it_remembers(make: Callable[[], Pooled]) -> None:
    """THE EQUALITY ORACLE: with nothing remembered, the wrapper IS the term, and after one step it is `across`.

    Any difference on the first step is the wrapper changing the loss for a run that has remembered
    nothing yet; any difference on the second is a memory written and then read as something else.
    """
    first, second = _step(SEED), _step(SEED + 1)
    remembered = Remembered(make(), capacity=4)

    assert torch.equal(remembered(first), make()(first))
    assert torch.equal(remembered(second), make().across(second, first))


@pytest.mark.parametrize("make", POOLED[:2])
def test_graded(make: Callable[[], Pooled]) -> None:
    """As a base, the memory widens the latent batch FIRST and the grade ladder spreads it LAST.

    Contrast is left out: the ladder spreads logits and targets, never the embedding, so it is not a base.
    """
    first, second = _step(SEED), _step(SEED + 1)
    ladder = OrdinalGrades()
    remembered = Remembered(make(), capacity=4)
    term = make()

    assert torch.equal(remembered.graded(first, ladder.spread), term(ladder.spread(first)))
    widened = term.widened(second, first)
    assert torch.equal(remembered.graded(second, ladder.spread), term(ladder.spread(widened)))
    assert torch.equal(Remembered(make()).graded(first, LabelStructure().spread), term(first))


def test_remembered_pools_a_repeated_batch_into_the_same_ratio():
    """Remembering the same batch and re-presenting it must leave the ranking ratio untouched.

    The three pair populations are copies, so summing numerators and denominators before dividing gives
    the same ratio — and ONLY if both cross directions are formed; dropping one moves it.
    """
    order = [0.4, -1.0, 2.0, 0.5, 3.0, -2.0]
    remembered = Remembered(Ranking())
    first = remembered(_batch(order))

    assert torch.allclose(remembered(_batch(order)), first, atol=1e-6)


def test_settings():
    """The pool is read off the memory, so a run's `--pair-memory` is checked against what would drop."""
    assert Remembered(Ranking(0.3), capacity=4).settings() == {"rank_weight": 0.3, PAIR_POOL: 4}
    assert Remembered(Margin(0.8), capacity=2).settings() == {"auc_margin": 0.8, PAIR_POOL: 2}


def test_attached():
    """The memory owns nothing; the wrapper answers whatever the wrapped term owns."""
    assert Remembered(Contrast()).attached(torch.device("cpu")) == []
