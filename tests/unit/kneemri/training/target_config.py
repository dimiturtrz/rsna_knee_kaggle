"""The label node: what a run is fitted to, and the one value the loss reads it through."""

import pytest

from core.data.labels.targets import CONFIDENCE_FLOOR, Supervision
from kneemri.training.config_tree import TrainConfig


def test_supervision(monkeypatch: pytest.MonkeyPatch):
    """The two label switches must reach the targets, from argv, without `main` in the way.

    Both change what the model is FITTED ON while every recorded config keeps agreeing with itself —
    the shape of nine defects here already — and they map to a data argument rather than to a policy
    object, so the `options_to_policies` walk cannot see them.
    """
    monkeypatch.setattr("sys.argv", ["train", "--labels", "labels.csv", "--weight-floor", "--calibrated-targets"])
    asked = TrainConfig.parse().targets.supervision

    assert asked.floor == CONFIDENCE_FLOOR, "a finding the report never mentions still pulls with this"
    assert asked.calibrated
    assert TrainConfig().targets.supervision == Supervision(), "unasked-for, the run trains on the raw levels"
