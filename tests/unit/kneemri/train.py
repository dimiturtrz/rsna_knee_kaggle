from pathlib import Path

import pytest

from kneemri.train import TrainingOptions


def test_parse(monkeypatch: pytest.MonkeyPatch):
    """Flags become a typed options object, so a run's configuration is inspectable without argv."""
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--labels", "labels.csv", "--fold", "3", "--epochs", "30", "--one-cycle", "--intensity-aug"],
    )
    options = TrainingOptions.parse()
    assert options.fold == 3
    assert options.epochs == 30
    assert options.one_cycle
    assert options.intensity_aug
    assert options.labels == Path("labels.csv")
    assert options.eval_labels is None, "no ruler given means score on what you trained on"


def test_parse_reads_a_separate_evaluation_ruler(monkeypatch: pytest.MonkeyPatch):
    """The ruler is separable, which is what makes two label arms comparable."""
    monkeypatch.setattr("sys.argv", ["train", "--labels", "a.csv", "--eval-labels", "b.csv"])
    options = TrainingOptions.parse()
    assert options.labels == Path("a.csv")
    assert options.eval_labels == Path("b.csv")
