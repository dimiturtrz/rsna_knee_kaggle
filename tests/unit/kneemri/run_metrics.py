import json
from pathlib import Path

from kneemri.run_metrics import EpochRow, RunMetrics


def test_record(tmp_path: Path):
    """Each epoch appends one JSON line beside the checkpoint, never overwriting the last."""
    metrics = RunMetrics(tmp_path / "member_fold0.pt")
    metrics.record(EpochRow(epoch=0, loss=1.2, auc=0.73, seconds=30, per_label={"ACL": 0.7}))
    metrics.record(EpochRow(epoch=1, loss=1.0, auc=0.79, seconds=31, per_label={"ACL": 0.8}))

    assert metrics.path.name == "member_fold0.metrics.jsonl", "it sits beside the member it describes"
    lines = metrics.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, "the CURVE is the point; a final score throws it away"
    assert json.loads(lines[1])["auc"] == 0.79


def test_rows(tmp_path: Path):
    """The record reads back as data, which is the whole reason it is not prose in a log."""
    metrics = RunMetrics(tmp_path / "member_fold0.pt")
    assert metrics.rows() == [], "a run that has not started yet is empty, not an error"

    metrics.record(EpochRow(epoch=0, loss=1.2, auc=0.73, seconds=30, per_label={"ACL": 0.7}))
    assert [row["epoch"] for row in metrics.rows()] == [0]


def test_best(tmp_path: Path):
    """The best scored epoch, not the last — that is the epoch whose weights the checkpoint holds.

    An unscored run (--fold -1 records NaN) must come back None rather than picking a NaN row: NaN
    compares false against everything, so an unguarded max() returns whichever row happened to be first.
    """
    checkpoint = tmp_path / "run.pt"
    empty: dict[str, float] = {}
    rows: list[EpochRow] = [
        {"epoch": 0, "loss": 1.0, "auc": 0.84, "seconds": 9, "per_label": empty},
        {"epoch": 1, "loss": 0.9, "auc": 0.81, "seconds": 9, "per_label": empty},
    ]
    checkpoint.with_suffix(".metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    best = RunMetrics(checkpoint).best()
    assert best is not None and best["epoch"] == 0

    unscored = tmp_path / "final.pt"
    unscored.with_suffix(".metrics.jsonl").write_text(
        json.dumps({"epoch": 0, "loss": 1.0, "auc": float("nan"), "seconds": 9, "per_label": {}}) + "\n",
        encoding="utf-8",
    )
    assert RunMetrics(unscored).best() is None


def test_exists(tmp_path: Path):
    """The recorder answers whether anything is there, so callers need not reach for its path.

    The distinction is worth making: a missing file is usually the wrong path, and an empty one is a run
    that died before its first scored epoch. They need different fixes.
    """
    metrics = RunMetrics(tmp_path / "run.pt")
    assert not metrics.exists()
    metrics.record(EpochRow(epoch=0, loss=1.0, auc=0.5, seconds=1, per_label={"ACL": 0.5}))
    assert metrics.exists()
