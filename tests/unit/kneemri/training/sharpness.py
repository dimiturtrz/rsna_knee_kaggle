from typing import cast

import torch
from torch import Tensor

from kneemri.training.sharpness import FLAT, Sharpness

RADIUS = 0.05
WIDTH = 3


def _weight(model: torch.nn.Module) -> Tensor:
    # `nn.Module.__getattr__` returns `Module | Tensor`, so a parameter read is untyped at every use
    # site. Narrowing it ONCE here is the alternative to thirteen suppressions that each say nothing.
    return cast("Tensor", model.weight)


def _model() -> torch.nn.Module:
    model = torch.nn.Linear(WIDTH, WIDTH, bias=False)
    with torch.no_grad():
        _weight(model).fill_(1.0)
    return model


def _with_gradient(model: torch.nn.Module, value: float) -> torch.nn.Module:
    weight = _weight(model)
    weight.grad = torch.full_like(weight, value)
    return model


def test_sharpness_climbed() -> None:
    model = _with_gradient(_model(), 2.0)
    moved = Sharpness(RADIUS).climbed(model)
    assert moved
    # Every entry of a constant gradient carries the same share of the norm, so the offset is the radius
    # spread evenly over the WIDTH*WIDTH entries: rho / sqrt(n), added to a weight of one.
    expected = 1.0 + RADIUS / (WIDTH * WIDTH) ** 0.5
    assert torch.allclose(_weight(model), torch.full_like(_weight(model), expected))


def test_sharpness_climbed_declines_when_flat() -> None:
    model = _with_gradient(_model(), 2.0)
    assert not Sharpness(FLAT).climbed(model)
    assert torch.allclose(_weight(model), torch.ones_like(_weight(model)))


def test_sharpness_climbed_declines_on_non_finite_gradient() -> None:
    model = _with_gradient(_model(), float("inf"))
    assert not Sharpness(RADIUS).climbed(model)
    assert torch.allclose(_weight(model), torch.ones_like(_weight(model)))


def test_sharpness_descended() -> None:
    model = _with_gradient(_model(), 2.0)
    sharpness = Sharpness(RADIUS)
    sharpness.climbed(model)
    # The second backward pass overwrites the gradient before the restore, which is exactly the case the
    # remembered offset exists for: recomputing the direction here would restore to the wrong point.
    _with_gradient(model, -7.0)
    sharpness.descended(model)
    assert torch.allclose(_weight(model), torch.ones_like(_weight(model)))


def test_sharpness_descended_is_a_no_op_without_a_climb() -> None:
    model = _with_gradient(_model(), 2.0)
    Sharpness(RADIUS).descended(model)
    assert torch.allclose(_weight(model), torch.ones_like(_weight(model)))


def test_sharpness_counts_what_it_did() -> None:
    """THE PLUMBING WITNESS: an arm whose ascent never fired is an ordinary run at half the throughput.

    Nothing downstream of the loop can tell the two apart -- the checkpoint, the score and the ledger row
    are identical -- so the counters are the only place the question is answerable, and the arm's own log
    is where they are read from.
    """
    sharpness = Sharpness(RADIUS)
    model = _model()
    _with_gradient(model, 1.0)
    assert sharpness.climbed(model)
    assert (sharpness.climbs, sharpness.declines) == (1, 0)
    sharpness.descended(model)

    _with_gradient(model, float("inf"))
    assert not sharpness.climbed(model)
    assert (sharpness.climbs, sharpness.declines) == (1, 1)
