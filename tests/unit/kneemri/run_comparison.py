import json
from pathlib import Path

import pytest

from kneemri.run_comparison import RunComparison
from kneemri.run_metrics import EpochRow

CONTROL: list[EpochRow] = [
    {"epoch": 0, "loss": 1.0, "auc": 0.80, "seconds": 10, "per_label": {"MCL": 0.70, "Effusion": 0.90}},
    {"epoch": 1, "loss": 0.9, "auc": 0.84, "seconds": 10, "per_label": {"MCL": 0.75, "Effusion": 0.93}},
]
ARM: list[EpochRow] = [
    {"epoch": 0, "loss": 1.0, "auc": 0.81, "seconds": 10, "per_label": {"MCL": 0.72, "Effusion": 0.90}},
    {"epoch": 1, "loss": 0.8, "auc": 0.86, "seconds": 10, "per_label": {"MCL": 0.85, "Effusion": 0.87}},
]


def _run(tmp_path: Path, name: str, rows: list[EpochRow]) -> Path:
    checkpoint = tmp_path / f"{name}.pt"
    checkpoint.with_suffix(".metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return checkpoint


def _describe(checkpoint: Path, recipe: dict[str, str]) -> None:
    """Prepend a recipe line, the way a real run writes one before its first epoch."""
    path = checkpoint.with_suffix(".metrics.jsonl")
    path.write_text(json.dumps({"recipe": recipe}) + "\n" + path.read_text(encoding="utf-8"), encoding="utf-8")


def test_deltas(tmp_path: Path):
    """Deltas are taken against the control's BEST epoch, per label, and the mean comes with them.

    The per-label split is the whole point: this arm gains 0.10 on MCL and LOSES 0.06 on Effusion,
    which is a concentrated gain — the shape the resolution and extended-slots arms predict — and a
    mean of +0.02 alone cannot tell that apart from a uniform lift.
    """
    comparison = RunComparison(_run(tmp_path, "ctl", CONTROL), [_run(tmp_path, "arm", ARM)])
    deltas = comparison.deltas()["arm.metrics"]
    assert deltas["MCL"] == pytest.approx(0.10)
    assert deltas["Effusion"] == pytest.approx(-0.06)
    assert deltas["MEAN"] == pytest.approx(0.02)


def test_deltas_refuses_a_control_with_no_scored_epoch(tmp_path: Path):
    """A final fit records NaN AUCs, so it can be an ARM but never the control it is measured against."""
    empty: dict[str, float] = {}
    unscored: list[EpochRow] = [{"epoch": 0, "loss": 1.0, "auc": float("nan"), "seconds": 10, "per_label": empty}]
    comparison = RunComparison(_run(tmp_path, "final", unscored), [_run(tmp_path, "arm", ARM)])
    with pytest.raises(ValueError, match="no scored epoch"):
        comparison.deltas()


def test_report(tmp_path: Path):
    """The report names the control's best epoch and orders labels by the size of the move."""
    comparison = RunComparison(_run(tmp_path, "ctl", CONTROL), [_run(tmp_path, "arm", ARM)])
    report = comparison.report()
    assert "0.8400" in report and "epoch 1" in report
    assert report.index("MCL") < report.index("Effusion"), "the widest delta is read first"


def test_report_names_what_else_moved(tmp_path: Path):
    """A delta is a claim about ONE axis. What the arm changed besides it decides whether that holds."""
    control = _run(tmp_path, "ctl", CONTROL)
    arm = _run(tmp_path, "arm", ARM)
    _describe(control, {"crop_mm": "130.0", "centre": "image", "epochs": "8"})
    _describe(arm, {"crop_mm": "80.0", "centre": "tissue", "epochs": "8"})

    report = RunComparison(control, [arm]).report()
    assert "crop_mm 130.0→80.0" in report
    assert "centre image→tissue" in report
    assert "epochs" not in report.split("vs control:")[1], "an option both runs share is not a difference"


def test_unmatched(tmp_path: Path):
    """Only the arm recorded it, so the control ran the default of the day — a question, not a delta.

    Kept apart because the two need different judgements: `train_size` was added after most of the
    ledger ran and is a no-op, while a recorded `backbone_lr` that disagrees is a second axis.
    """
    control = _run(tmp_path, "ctl", CONTROL)
    arm = _run(tmp_path, "arm", ARM)
    _describe(control, {"crop_mm": "130.0"})
    _describe(arm, {"crop_mm": "80.0", "train_size": "0"})

    comparison = RunComparison(control, [arm])
    said = comparison.unmatched(comparison.arms[0])

    assert "crop_mm 130.0→80.0" in said
    assert "control predates train_size=0" in said
    assert "unset" not in said, "an absence is not a value the control held"


def test_unmatched_says_so_when_a_run_recorded_no_recipe(tmp_path: Path):
    """Silence read as a match is the failure; every file predating the recipe is in that state."""
    report = RunComparison(_run(tmp_path, "ctl", CONTROL), [_run(tmp_path, "arm", ARM)]).report()
    assert "recipe unrecorded" in report
