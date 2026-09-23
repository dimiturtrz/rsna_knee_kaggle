import json
from pathlib import Path

import pytest
import torch

from kneemri.runs.run_metrics import BYTES_PER_GB, RunMetrics
from kneemri.runs.run_record import EpochRow, StepRow

RESERVED_GB = 26.0
ALLOCATED_GB = 23.33


def test_record(tmp_path: Path):
    """Each epoch appends one JSON line beside the checkpoint, never overwriting the last."""
    metrics = RunMetrics(tmp_path / "member_fold0.pt")
    metrics.record(EpochRow(epoch=0, loss=1.2, auc=0.73, seconds=30, per_label={"ACL": 0.7}))
    metrics.record(EpochRow(epoch=1, loss=1.0, auc=0.79, seconds=31, per_label={"ACL": 0.8}))

    assert metrics.path.name == "member_fold0.metrics.jsonl", "it sits beside the member it describes"
    lines = metrics.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, "the CURVE is the point; a final score throws it away"
    assert json.loads(lines[1])["auc"] == 0.79


def test_record_attaches_reserved_as_the_peak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The peak the cap is compared against is RESERVED; allocated is recorded beside it, not as it.

    `--vram-gb` bounds what the allocator reserves, so a row carrying allocated under `vram_peak_gb`
    reports headroom a run does not have — the gap between the two is exactly the fragmentation that
    wrote the 448 px OOM. Faked stats rather than a card: this asserts which reading lands in which
    field, which is the part that was wrong, and it must hold on a machine with no GPU at all.
    """
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_: RESERVED_GB * BYTES_PER_GB)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: ALLOCATED_GB * BYTES_PER_GB)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    metrics = RunMetrics(tmp_path / "capped.pt")

    metrics.record(EpochRow(epoch=0, loss=1.3, auc=0.65, seconds=61, per_label={"ACL": 0.7}))

    recorded = metrics.rows()[0]
    assert recorded["vram_peak_gb"] == RESERVED_GB
    assert recorded["vram_alloc_gb"] == ALLOCATED_GB


def test_begin(tmp_path: Path):
    """A re-run under the same name opens its own record; the previous run's epochs move aside.

    The failure it removes is `slices16_ws_s2026`: a finished run peaking at 0.8452 followed by a
    killed one peaking at 0.8411 in ONE file, whose checkpoint belongs to the finished run — so every
    reader of `best` quoted the killed run's number for a member that scores four thousandths higher.
    """
    metrics = RunMetrics(tmp_path / "run.pt")
    metrics.begin({"slices": "12"})
    metrics.record(EpochRow(epoch=0, loss=1.0, auc=0.8452, seconds=1, per_label={"ACL": 0.5}))

    metrics.begin({"slices": "16"})
    metrics.record(EpochRow(epoch=0, loss=1.0, auc=0.8411, seconds=1, per_label={"ACL": 0.5}))

    assert metrics.best() == {**metrics.rows()[0]}, "the live file holds this run alone"
    assert [row["auc"] for row in metrics.rows()] == [0.8411]
    assert metrics.recipe() == {"slices": "16"}
    aside = (tmp_path / "run.metrics.1.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["auc"] for line in aside if "auc" in line] == [0.8452], (
        "the earlier run is kept beside the new one, not overwritten"
    )

    metrics.begin({"slices": "20"})
    assert (tmp_path / "run.metrics.2.jsonl").exists(), "a third run rotates to its own number"


def test_begin_rotates_the_artefacts_with_the_epochs(tmp_path: Path):
    """The weights and the predictions move aside under the SAME index as the epochs they belong to.

    Rotating the epochs alone is what cost `depth_regions_span` its run-1 predictions on 2026-08-19: it
    re-entered `run_train` mid-flight, trained again under its own name, and the ledger's row went on
    quoting a best whose `.holdout.npz` had been overwritten under it. One index across the three files
    is the whole point — an account without the weights it describes cannot be re-scored.
    """
    metrics = RunMetrics(tmp_path / "run.pt")
    metrics.begin({"slices": "12"})
    metrics.record(EpochRow(epoch=0, loss=1.0, auc=0.8505, seconds=1, per_label={"ACL": 0.5}))
    (tmp_path / "run.pt").write_text("first weights", encoding="utf-8")
    (tmp_path / "run.holdout.npz").write_text("first predictions", encoding="utf-8")

    metrics.begin({"slices": "12"})

    assert (tmp_path / "run_run1.pt").read_text(encoding="utf-8") == "first weights"
    assert (tmp_path / "run_run1.holdout.npz").read_text(encoding="utf-8") == "first predictions"
    assert not (tmp_path / "run.pt").exists(), "the live name is free for the run that is starting"
    assert (tmp_path / "run.metrics.1.jsonl").exists(), "and its epochs carry the same index"


def test_begin_carries_the_predictions_back_when_resuming(tmp_path: Path):
    """A resuming run keeps the rotated reading under the live name; the aside copy stays whole.

    `--resume` restores the KEPT epoch, and a predictions file is rewritten only when a new best
    fires — so a resume onto a best no later epoch beats used to finish with its declared
    `# reading:` path absent. `labelcut_v3_ema_s2026` did exactly that: e23 done, 0.9039 sitting
    under `_run1`, `reading_check` refusing an arm that had succeeded.
    """
    metrics = RunMetrics(tmp_path / "run.pt")
    metrics.begin({"slices": "12"})
    metrics.record(EpochRow(epoch=0, loss=1.0, auc=0.8668, seconds=1, per_label={"ACL": 0.5}))
    (tmp_path / "run.holdout.npz").write_text("kept predictions", encoding="utf-8")

    metrics.begin({"slices": "12"}, resuming=True)

    assert (tmp_path / "run.holdout.npz").read_text(encoding="utf-8") == "kept predictions", (
        "the reading of the checkpoint the resume starts from is under the live name"
    )
    assert (tmp_path / "run_run1.holdout.npz").read_text(encoding="utf-8") == "kept predictions", (
        "copied, not moved — the rotated run keeps its own record beside its epochs"
    )


def test_begin_leaves_the_live_name_free_when_not_resuming(tmp_path: Path):
    """A plain re-run is a DIFFERENT run, so it may not inherit the previous one's reading."""
    metrics = RunMetrics(tmp_path / "run.pt")
    metrics.begin({"slices": "12"})
    metrics.record(EpochRow(epoch=0, loss=1.0, auc=0.84, seconds=1, per_label={"ACL": 0.5}))
    (tmp_path / "run.holdout.npz").write_text("first predictions", encoding="utf-8")

    metrics.begin({"slices": "16"})

    assert not (tmp_path / "run.holdout.npz").exists()


def test_begin_rotates_what_a_killed_run_left(tmp_path: Path):
    """A run that scored an epoch but never checkpointed still rotates, and takes no absent file with it.

    The rotation number comes from the epochs because they are the one sibling every run writes; a
    missing checkpoint must not shift the index and must not raise.
    """
    metrics = RunMetrics(tmp_path / "run.pt")
    metrics.begin({"slices": "12"})
    metrics.record(EpochRow(epoch=0, loss=1.0, auc=0.84, seconds=1, per_label={"ACL": 0.5}))

    metrics.begin({"slices": "12"})

    assert (tmp_path / "run.metrics.1.jsonl").exists()
    assert not (tmp_path / "run_run1.pt").exists(), "nothing is invented for a run that never checkpointed"


def test_begin_leaves_a_first_run_where_it_is(tmp_path: Path):
    """Nothing to move aside means no rotation — an empty file is not a run that happened."""
    metrics = RunMetrics(tmp_path / "run.pt")

    metrics.begin({"slices": "12"})
    metrics.begin({"slices": "16"})

    assert not (tmp_path / "run.metrics.1.jsonl").exists(), "a described-but-unscored run is not rotated"
    assert metrics.recipe() == {"slices": "16"}


def test_describe(tmp_path: Path):
    """The recipe shares the file with the scores, so every reader of the scores must ignore it."""
    metrics = RunMetrics(tmp_path / "run.pt")

    metrics.describe({"crop_mm": "80.0", "centre": "tissue"})
    metrics.record(EpochRow(epoch=0, loss=1.0, auc=0.5, seconds=1, per_label={"ACL": 0.5}))

    assert [row["epoch"] for row in metrics.rows()] == [0], "the recipe line is not an epoch"
    assert metrics.best() is not None, "and ranking epochs must not trip over it"


def test_steps_taken(tmp_path: Path):
    """Each epoch's steps APPEND, so one stream holds the whole run."""
    metrics = RunMetrics(tmp_path / "run.pt")
    metrics.steps_taken([StepRow(epoch=0, step=0, loss=0.7, grad_norm=3.0, lr=[1e-4])])
    metrics.steps_taken([StepRow(epoch=1, step=1, loss=0.5, grad_norm=2.0, lr=[5e-5])])

    lines = (tmp_path / "run.steps.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["epoch"] for line in lines] == [0, 1]


def test_begin_rotates_every_file_of_the_run(tmp_path: Path):
    """The step stream, the draws and the per-epoch predictions rotate under the epochs' index too."""
    metrics = RunMetrics(tmp_path / "run.pt")
    metrics.begin({"slices": "12"})
    metrics.record(EpochRow(epoch=0, loss=1.0, auc=0.8, seconds=1, per_label={"ACL": 0.8}))
    for spelling in (".steps.jsonl", ".draws.npz", ".epochs.npz"):
        (tmp_path / f"run{spelling}").write_text("first run", encoding="utf-8")

    metrics.begin({"slices": "12"})

    for spelling in (".steps.jsonl", ".draws.npz", ".epochs.npz"):
        assert (tmp_path / f"run_run1{spelling}").read_text(encoding="utf-8") == "first run"
        assert not (tmp_path / f"run{spelling}").exists()
