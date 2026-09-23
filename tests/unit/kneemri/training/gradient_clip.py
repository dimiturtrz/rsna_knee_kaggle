"""The ceiling on one optimiser step's gradient norm, and the null policy that is the default."""

import pytest
import torch

from kneemri.training.gradient_clip import UNCLIPPED, GradientClip


def _model_with_gradient(norm: float) -> torch.nn.Module:
    """One parameter carrying a gradient of exactly `norm`, so the clip's arithmetic is readable."""
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.grad = torch.tensor([[norm]])
    return model


def test_applied():
    """A gradient above the ceiling is rescaled ONTO it; one below is left alone; off touches nothing.

    The direction must survive — clipping is a rescale of the whole gradient, not a clamp per element —
    so the sign is asserted beside the magnitude. `UNCLIPPED` is the state every number in the ledger
    was measured under, and it has to be indistinguishable from not calling this at all.
    """
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    over = _model_with_gradient(-4.0)
    optimiser = torch.optim.SGD(over.parameters(), lr=0.0)
    norm = GradientClip(2.0).applied(over, optimiser, scaler)
    assert float(norm) == pytest.approx(4.0), "the norm returned is the one BEFORE clipping"
    # APPROX, because `clip_grad_norm_` divides by the norm plus a 1e-6 epsilon rather than by the norm,
    # so the clipped value lands a millionth under the ceiling by construction and never on it.
    assert float(over.weight.grad) == pytest.approx(-2.0), "a norm of 4 under a ceiling of 2 halves, keeping its sign"

    under = _model_with_gradient(0.5)
    GradientClip(2.0).applied(under, torch.optim.SGD(under.parameters(), lr=0.0), scaler)
    assert float(under.weight.grad) == 0.5, "below the ceiling the gradient is untouched"

    off = _model_with_gradient(9.0)
    unclipped = GradientClip(UNCLIPPED).applied(off, torch.optim.SGD(off.parameters(), lr=0.0), scaler)
    assert float(unclipped) == pytest.approx(9.0), "measured even when nothing is clipped"
    assert float(off.weight.grad) == 9.0, "the default must be the unbounded step the ledger was measured on"
