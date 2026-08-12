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
    unscored: list[EpochRow] = [
        {"epoch": 0, "loss": 1.0, "auc": float("nan"), "seconds": 10, "per_label": empty}
    ]
    comparison = RunComparison(_run(tmp_path, "final", unscored), [_run(tmp_path, "arm", ARM)])
    with pytest.raises(ValueError, match="no scored epoch"):
        comparison.deltas()


def test_report(tmp_path: Path):
    """The report names the control's best epoch and orders labels by the size of the move."""
    comparison = RunComparison(_run(tmp_path, "ctl", CONTROL), [_run(tmp_path, "arm", ARM)])
    report = comparison.report()
    assert "0.8400" in report and "epoch 1" in report
    assert report.index("MCL") < report.index("Effusion"), "the widest delta is read first"
