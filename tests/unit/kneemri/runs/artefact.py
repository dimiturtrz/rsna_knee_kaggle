from pathlib import Path

import pytest

from kneemri.runs.artefact import Artefact


@pytest.mark.parametrize(
    "held",
    ["run.pt", "run.metrics.jsonl", "run.holdout.npz", "run.steps.jsonl", "run.draws.npz", "run.partial", "run"],
)
def test_stem(held: str):
    """Every sibling, and the bare stem, names the same run — `.metrics.jsonl` never reads as `run.metrics`."""
    assert Artefact.stem(Path("d") / held) == "run"


def test_beside(tmp_path: Path):
    """Any sibling reaches any other, which `with_suffix` cannot do across a two-part spelling."""
    assert Artefact.CHECKPOINT.beside(tmp_path / "run.metrics.jsonl") == tmp_path / "run.pt"
    assert Artefact.METRICS.beside(tmp_path / "run.pt") == tmp_path / "run.metrics.jsonl"
    assert Artefact.STEPS.beside(tmp_path / "run.holdout.npz") == tmp_path / "run.steps.jsonl"


def test_aside(tmp_path: Path):
    """An earlier run's files go under `_run<n>`, except the record, which keeps its on-disk numbering."""
    assert Artefact.CHECKPOINT.aside(tmp_path / "run.pt", 2) == tmp_path / "run_run2.pt"
    assert Artefact.HOLDOUT.aside(tmp_path / "run.metrics.jsonl", 1) == tmp_path / "run_run1.holdout.npz"
    assert Artefact.METRICS.aside(tmp_path / "run.pt", 3) == tmp_path / "run.metrics.3.jsonl"
