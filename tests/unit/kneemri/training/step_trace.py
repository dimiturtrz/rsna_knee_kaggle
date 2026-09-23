"""One epoch's optimiser steps, kept on the device and read back once."""

import numpy as np
import pytest
import torch

from kneemri.training.grades import Predicted
from kneemri.training.loss_quantiles import LOSS_QUANTILES
from kneemri.training.step_trace import StepTrace


def _judged(logits: list[list[float]], scores: list[list[float]]) -> Predicted:
    logit = torch.tensor(logits)
    return Predicted(logits=logit, scores=torch.tensor(scores), weights=torch.ones_like(logit))


def _optimiser(rate: float) -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=rate)


def _stepped(trace: StepTrace, *norms: float) -> StepTrace:
    for step, norm in enumerate(norms):
        trace.stepped(torch.tensor(float(step)), torch.tensor(norm), _optimiser(0.1 * (step + 1)))
    return trace


def test_opened():
    """A new epoch keeps what was drawn and forgets the last epoch's steps."""
    trace = _stepped(StepTrace(), 1.0)
    trace.opened(np.array([4, 4, 2], dtype=np.int64))
    assert trace.drawn.tolist() == [4, 4, 2]
    assert trace.rows(0) == [], "no step of an earlier epoch may leak into this one"


def test_judged():
    """A study's loss is its weighted BCE averaged over its labels; the quantiles are read off them."""
    trace = StepTrace()
    trace.judged(_judged([[0.0, 0.0]], [[1.0, 0.0]]))
    trace.closed()
    assert trace.quantiles["q50"] == pytest.approx(np.log(2)), "a zero logit costs log 2 whatever the truth"


def test_stepped():
    """Each step keeps its loss, its norm, and the rate of every parameter group."""
    rows = _stepped(StepTrace(), 3.0).rows(5)
    assert rows == [{"epoch": 5, "step": 0, "loss": 0.0, "grad_norm": 3.0, "lr": [pytest.approx(0.1)]}]


def test_closed():
    """Closing an epoch with no studies still answers with every quantile key, as NaN."""
    trace = StepTrace()
    assert trace.quantiles == {}, "nothing has been fitted, so there is no distribution yet"
    trace.closed()
    assert tuple(trace.quantiles) == LOSS_QUANTILES
    assert all(np.isnan(value) for value in trace.quantiles.values())


def test_rows():
    """Steps come back in the order they were taken, numbered from zero within the epoch."""
    rows = _stepped(StepTrace(), 1.0, 2.0).rows(0)
    assert [row["step"] for row in rows] == [0, 1]
    assert [row["grad_norm"] for row in rows] == [1.0, 2.0]


def test_optimised():
    """Norms summarise over FINITE steps; the non-finite ones are counted instead of poisoning the mean."""
    summary = _stepped(StepTrace(), 1.0, 3.0, float("inf")).optimised()
    assert summary["grad_norm_mean"] == pytest.approx(2.0)
    assert summary["grad_norm_max"] == pytest.approx(3.0)
    assert summary["nonfinite_steps"] == 1
    assert summary["lr"] == [pytest.approx(0.3)], "the rate the epoch ENDED on"
    assert np.isnan(StepTrace().optimised()["grad_norm_mean"]), "no steps is unread, not zero"


def test_fit_auc():
    """A perfect ranking of the studies trained on reads 1; a label with one class is left out."""
    trace = StepTrace()
    assert np.isnan(trace.fit_auc()), "nothing judged yet"
    trace.judged(_judged([[2.0, 0.0], [-2.0, 0.0]], [[1.0, 1.0], [0.0, 1.0]]))
    assert trace.fit_auc() == pytest.approx(1.0)


def test_loss_fields():
    """The quantiles under the names the epoch row carries them by."""
    trace = StepTrace()
    trace.quantiles = {"q50": 0.3}
    assert trace.loss_fields() == {"loss_q50": 0.3}
