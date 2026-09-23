from pathlib import Path

import torch

from kneemri.experiments.ledger_check import LedgerCheck
from kneemri.experiments.runs_table import RunsTable
from kneemri.runs.run_metrics import RunMetrics
from kneemri.runs.run_record import EpochRow


def _table(tmp_path: Path) -> RunsTable:
    models = tmp_path / "models"
    models.mkdir()
    ledger = tmp_path / "README.md"
    ledger.write_text("`recorded_arm` scored well", encoding="utf-8")
    recorded = RunMetrics(models / "recorded_arm_s2026.pt")
    recorded.begin({"fold": "0"}, run={"id": "run-a", "git": "abc"})
    recorded.record(EpochRow(epoch=0, loss=1.0, auc=0.8, seconds=1, per_label={"ACL": 0.7}))
    recorded.record(EpochRow(epoch=1, loss=0.9, auc=0.85, seconds=1, per_label={"ACL": 0.8}))
    torch.save({"state": {}}, models / "recorded_arm_s2026.pt")
    torch.save({"state": {}, "run": {"id": "run-b"}}, models / "stamped_only.pt")
    torch.save({"state": {}}, models / "bare_old.pt")
    return RunsTable(models, LedgerCheck(ledger))


def test_names(tmp_path: Path):
    """A checkpoint or a live record names a run; a rotated record does not name a second one."""
    table = _table(tmp_path)
    (table.models / "recorded_arm_s2026.metrics.1.jsonl").write_text("", encoding="utf-8")

    assert table.names() == ["bare_old", "recorded_arm_s2026", "stamped_only"]


def test_row(tmp_path: Path):
    """Each cell from the file that states it; a fact nobody recorded stays an empty cell."""
    table = _table(tmp_path)

    recorded = table.row("recorded_arm_s2026")
    assert recorded["run_id"] == "run-a"
    assert recorded["git"] == "abc"
    assert (recorded["epochs"], recorded["best_epoch"], recorded["best_auc"]) == ("2", "1", "0.8500")
    assert (recorded["checkpoint"], recorded["holdout"], recorded["ledgered"]) == ("yes", "no", "yes")

    assert table.row("stamped_only")["run_id"] == "run-b", "a copy's id travels in the weights"
    bare = table.row("bare_old")
    assert bare["run_id"] == bare["git"] == bare["best_auc"] == bare["epochs"] == "", "blank, never guessed"
    assert bare["ledgered"] == "no"


def test_write(tmp_path: Path):
    """One header and one line per run, tab-separated."""
    table = _table(tmp_path)
    out = tmp_path / "runs.tsv"

    assert table.write(out) == 3
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t")[0] == "name"
    assert len(lines) == 4
