"""The weight average, on a model small enough to check by hand."""

import torch
from torch import nn

from kneemri.ema import WeightAverage

SEED = 2026


def _model(value: float) -> nn.Module:
    """A one-parameter linear layer whose weight is exactly `value`."""
    layer = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        layer.weight.fill_(value)
    return layer


def test_update():
    """One step moves the average toward the live weights, and never past them."""
    model = _model(0.0)
    average = WeightAverage(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(1.0)
    average.update(model)

    moved = float(average.shadow["weight"])
    assert 0.0 < moved < 1.0


def test_update_ramps_in_so_the_initialisation_does_not_dominate():
    """The first step must follow the live weights closely rather than averaging over random init.

    With a flat 0.999 decay the average would still be 99.9% initialisation after one step, which is
    the one set of weights known to be worthless.
    """
    model = _model(0.0)
    average = WeightAverage(model, decay=0.999)
    with torch.no_grad():
        model.weight.fill_(1.0)
    average.update(model)

    assert float(average.shadow["weight"]) > 0.5


def test_state_dict():
    """Averaged floats come back in the model's own dtype, and every key is present."""
    model = _model(1.0)
    average = WeightAverage(model)
    state = average.state_dict(model)

    assert set(state) == set(model.state_dict())
    assert state["weight"].dtype == model.state_dict()["weight"].dtype


def test_applied():
    """Scoring under the averaged weights must not leave them behind — training continues from live."""
    model = _model(0.0)
    average = WeightAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(1.0)
    average.update(model)

    with average.applied(model):
        inside = float(model.weight)
    assert inside != 1.0
    assert float(model.weight) == 1.0
