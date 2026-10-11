"""The epoch row as the union of what each probe reads."""

import numpy as np
import pytest
import torch

from core.findings import TARGETS
from kneemri.training.epoch_probes import (
    PROBES,
    ClockProbe,
    DrawProbe,
    EpochReading,
    FitProbe,
    HoldoutCells,
    HoldoutProbe,
    LossProbe,
    OptimProbe,
)
from kneemri.training.step_trace import StepTrace


def _reading(*, predicted: np.ndarray | None) -> EpochReading:
    trace = StepTrace()
    trace.opened(np.array([0, 0, 1], dtype=np.int64))
    trace.stepped(torch.tensor(0.5), torch.tensor(2.0), torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01))
    trace.closed()
    drawn_scores = np.zeros((3, len(TARGETS)), dtype=np.float32)
    drawn_scores[:, 0] = [1.0, 1.0, 0.0]
    return EpochReading(
        epoch=2,
        loss=0.7,
        started=100.4,
        fit_seconds=8.6,
        seconds=10.2,
        ranked=0.8,
        per_label={"ACL": 0.8},
        predicted=predicted,
        truth=np.array([[1.0], [0.0]], dtype=np.float32),
        drawn=trace.drawn,
        drawn_scores=drawn_scores,
        trace=trace,
    )


def test_clock_probe_read():
    """Whole seconds, and the fitting share kept apart from the total."""
    reading = _reading(predicted=None)
    assert ClockProbe().read(reading) == {"epoch": 2, "started": 100, "seconds": 10, "fit_seconds": 9}


def test_loss_probe_read():
    """The mean loss, plus a quantile that reads NaN when no study was judged."""
    read = LossProbe().read(_reading(predicted=None))
    assert read["loss"] == pytest.approx(0.7)
    assert np.isnan(read["loss_q50"]), "no study was judged this epoch"


def test_holdout_probe_read():
    """A perfect holdout is calibrated with zero loss."""
    held = HoldoutProbe().read(_reading(predicted=np.array([[1.0], [0.0]])))
    assert held["auc"] == pytest.approx(0.8)
    assert held["ece"] == pytest.approx(0.0)
    assert held["holdout_loss"] == pytest.approx(0.0, abs=1e-5)


def test_optim_probe_read():
    """The one step's pre-clip norm is both the mean and the max."""
    read = OptimProbe().read(_reading(predicted=None))
    assert read["grad_norm_max"] == pytest.approx(2.0)
    assert read["nonfinite_steps"] == 0


def test_fit_probe_read():
    """No judged micro-batch, so no fit reading."""
    assert np.isnan(FitProbe().read(_reading(predicted=None))["fit_auc"])


def test_draw_probe_read():
    """A repeated draw is one study and two positives."""
    drawn = DrawProbe().read(_reading(predicted=None))
    assert (drawn["drawn_rows"], drawn["drawn_studies"]) == (3, 2), "a repeated draw is one study"
    assert drawn["drawn_positives"][TARGETS[0]] == 2


def test_calibration_error():
    """Confident and right is calibrated; confident and wrong is off by the whole confidence."""
    truth = np.array([[1.0], [0.0]], dtype=np.float32)
    assert HoldoutCells(np.array([[1.0], [0.0]]), truth).calibration_error() == pytest.approx(0.0)
    assert HoldoutCells(np.array([[0.0], [1.0]]), truth).calibration_error() == pytest.approx(1.0)
    assert np.isnan(HoldoutCells(np.array([[np.nan]]), truth[:1]).calibration_error())


def test_loss():
    """BCE against GRADED truth, NaN cells excluded."""
    truth = np.array([[0.5, np.nan]], dtype=np.float32)
    assert HoldoutCells(np.array([[0.5, 0.9]]), truth).loss() == pytest.approx(np.log(2))
    assert np.isnan(HoldoutCells(np.array([[np.nan]]), truth[:, :1]).loss())


def test_row():
    """The row is the union of every probe, and an unscored epoch still carries every holdout key."""
    row = _reading(predicted=None).row(PROBES)
    for probe in PROBES:
        assert set(probe.read(_reading(predicted=None))) <= set(row)
    assert np.isnan(row["ece"]), "nothing held out reads NaN, matching the `auc` column of a final fit"
    assert np.isnan(row["holdout_loss"])
