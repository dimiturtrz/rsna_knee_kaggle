from pathlib import Path

import torch

from kneemri.evaluation.metrics_recipe import MetricsRecipe
from kneemri.evaluation.run_metrics import EpochRow, RunMetrics


def _run(root: Path, name: str, *, options: dict[str, object] | None) -> RunMetrics:
    metrics = RunMetrics(root / f"{name}.pt")
    metrics.record(EpochRow(epoch=0, loss=1.0, auc=0.8, seconds=9, per_label={"ACL": 0.8}))
    if options is not None:
        torch.save({"options": options, "holdout": 0.8}, root / f"{name}.pt")
    return metrics


def test_checkpoint_of():
    """`.metrics.jsonl` is TWO suffixes, so the obvious `with_suffix` would produce `run.metrics.pt`."""
    assert MetricsRecipe.checkpoint_of(Path("d/run.metrics.jsonl")).name == "run.pt"


def test_backfill(tmp_path: Path):
    """The knowledge was always on disk — in the member, not beside the numbers that get compared."""
    metrics = _run(tmp_path, "arm", options={"crop_mm": 80.0, "centre": "tissue"})
    assert metrics.recipe() is None

    filled, orphaned = MetricsRecipe(tmp_path).backfill(write=True)

    assert (filled, orphaned) == (["arm.metrics.jsonl"], [])
    assert metrics.recipe() == {"crop_mm": "80.0", "centre": "tissue"}
    assert [row["epoch"] for row in metrics.rows()] == [0], "the scores it already held are untouched"


def test_backfill_names_a_run_whose_checkpoint_is_gone_rather_than_guessing(tmp_path: Path):
    """A deleted member cannot be a matched control again, and saying so beats inventing a recipe."""
    _run(tmp_path, "lost", options=None)

    assert MetricsRecipe(tmp_path).backfill(write=True) == ([], ["lost.metrics.jsonl"])


def test_backfill_under_check_writes_nothing(tmp_path: Path):
    """The reporting form has to leave the ledger exactly as it found it."""
    metrics = _run(tmp_path, "arm", options={"epochs": 8})

    filled, _ = MetricsRecipe(tmp_path).backfill(write=False)

    assert filled == ["arm.metrics.jsonl"]
    assert metrics.recipe() is None


def test_recipe_of(tmp_path: Path):
    """The oldest members predate the options stamp too; that is an absence, not an empty recipe."""
    torch.save({"holdout": 0.8}, tmp_path / "old.pt")

    assert MetricsRecipe.recipe_of(tmp_path / "old.pt") is None
