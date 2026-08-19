import json
from pathlib import Path

import pytest
import torch

from kneemri.runs.run_metrics import BYTES_PER_GB, EpochRow, RunMetrics

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


def test_rows(tmp_path: Path):
    """The record reads back as data, which is the whole reason it is not prose in a log."""
    metrics = RunMetrics(tmp_path / "member_fold0.pt")
    assert metrics.rows() == [], "a run that has not started yet is empty, not an error"

    metrics.record(EpochRow(epoch=0, loss=1.2, auc=0.73, seconds=30, per_label={"ACL": 0.7}))
    assert [row["epoch"] for row in metrics.rows()] == [0]


def test_rows_answers_for_the_run_the_recipe_describes(tmp_path: Path):
    """A re-run to the same --out appends, and only the newest run's checkpoint survives beside it.

    This is `pair_memory_s2026`: twelve epochs peaking at 0.8353, then four more peaking at 0.8155.
    Reading both runs' rows quotes 0.8353 for a checkpoint that no longer exists — and its kept
    predictions, which are `trained` and so exact, score 0.8155.
    """
    metrics = RunMetrics(tmp_path / "rerun.pt")
    metrics.describe({"fold": "0", "epochs": "12"})
    for epoch, auc in enumerate([0.73, 0.83, 0.81]):
        metrics.record(EpochRow(epoch=epoch, loss=1.0, auc=auc, seconds=233, per_label={}))

    metrics.describe({"fold": "0", "epochs": "12"})
    for epoch, auc in enumerate([0.71, 0.79]):
        metrics.record(EpochRow(epoch=epoch, loss=1.0, auc=auc, seconds=233, per_label={}))

    assert [row["auc"] for row in metrics.rows()] == [0.71, 0.79], "the run the file's recipe describes"
    best = metrics.best()
    assert best is not None and best["auc"] == 0.79, "not the 0.83 belonging to a replaced checkpoint"


def test_rows_reads_a_run_whose_recipe_was_back_filled(tmp_path: Path):
    """A recipe does not always precede its epochs, so position cannot say which run a row belongs to.

    Six files on disk carry theirs appended AFTER the run finished. Splitting at the last recipe line
    reports every one of them as having recorded nothing — the same run, erased instead of misread.
    """
    metrics = RunMetrics(tmp_path / "backfilled.pt")
    for epoch, auc in enumerate([0.73, 0.83]):
        metrics.record(EpochRow(epoch=epoch, loss=1.0, auc=auc, seconds=233, per_label={}))
    metrics.describe({"fold": "0", "epochs": "2"})

    assert [row["auc"] for row in metrics.rows()] == [0.73, 0.83], "described late is still described"


def test_epoch_seconds(tmp_path: Path):
    """One stalled epoch must not move the cost of a run, because on disk one already does.

    The shape is `pair_memory_s2026`'s: eleven epochs near 233 s and a single 27,266 s stall. The mean
    reads 2,690 s and would report the arm as 8x its control's cost; the median reads its real one.
    """
    metrics = RunMetrics(tmp_path / "stalled.pt")
    assert metrics.epoch_seconds() == 0.0, "a run that recorded nothing has no cost, not a crash"

    empty: dict[str, float] = {}
    for epoch, seconds in enumerate([233, 234, 233, 27266, 235, 233, 234]):
        metrics.record(EpochRow(epoch=epoch, loss=1.0, auc=0.8, seconds=seconds, per_label=empty))

    assert metrics.epoch_seconds() == 234


def test_stalls(tmp_path: Path):
    """The median made the stall harmless; this makes it visible, which is the other half.

    Same shape on disk as `pair_memory_s2026`. The run must name the 27,266 s epoch and NOTHING else —
    a per-epoch spread of 233-235 s is what a normal run looks like, and a rule that called any of those
    a stall would report one on every run in the directory.
    """
    metrics = RunMetrics(tmp_path / "stalled.pt")
    assert metrics.stalls() == [], "a run that recorded nothing has no stall, not a crash"

    empty: dict[str, float] = {}
    for epoch, seconds in enumerate([233, 234, 233, 27266, 235, 233, 234]):
        metrics.record(
            EpochRow(epoch=epoch, loss=1.0, auc=0.8, seconds=seconds, per_label=empty, started=1_760_000_000 + epoch)
        )

    stalls = metrics.stalls()
    assert [row["epoch"] for row in stalls] == [3]
    assert stalls[0]["started"] == 1_760_000_003, "WHEN is what lines a stall up against a suspend"


def test_stalls_needs_enough_epochs_to_have_a_median(tmp_path: Path):
    """With two rows the median sits between them, so the slower one is always above any factor of it.

    A first epoch that pays a warm-up would then report itself as a stall on every run that exists —
    a detector that fires on everything says the same thing as one that fires on nothing.
    """
    metrics = RunMetrics(tmp_path / "short.pt")
    empty: dict[str, float] = {}
    for epoch, seconds in enumerate([600, 100]):
        metrics.record(EpochRow(epoch=epoch, loss=1.0, auc=0.8, seconds=seconds, per_label=empty))

    assert metrics.stalls() == []


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

    # The rows a run WROTE can be wrong while the recipe stays right. Until 2026-08-16 an unscored run
    # recorded the epoch INDEX in this column, and two such files sit on disk: `max()` reads a
    # twelve-epoch final fit as scoring 11.0, which beats every real member in the ledger. The recipe
    # says the run held nothing back, and that outranks any number in the rows.
    indexed = tmp_path / "indexed.pt"
    indexed.with_suffix(".metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"recipe": {"fold": "-1", "epochs": "12"}}),
                *(
                    json.dumps({"epoch": e, "loss": 1.0, "auc": float(e), "seconds": 9, "per_label": {}})
                    for e in range(3)
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert RunMetrics(indexed).best() is None, "an epoch index is not a score, whatever column it sits in"


def test_exists(tmp_path: Path):
    """The recorder answers whether anything is there, so callers need not reach for its path.

    The distinction is worth making: a missing file is usually the wrong path, and an empty one is a run
    that died before its first scored epoch. They need different fixes.
    """
    metrics = RunMetrics(tmp_path / "run.pt")
    assert not metrics.exists()
    metrics.record(EpochRow(epoch=0, loss=1.0, auc=0.5, seconds=1, per_label={"ACL": 0.5}))
    assert metrics.exists()


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


def test_checkpoint(tmp_path: Path):
    """`.metrics.jsonl` is TWO suffixes, so the obvious `with_suffix` yields `run.metrics.pt`.

    That path is one nothing writes and everything then reads as a run carrying no stamp — which is
    how a checkpoint holding a full recipe reads as a run that recorded none. Both spellings of a run
    resolve to the same pair, because both reach this class.
    """
    assert RunMetrics(tmp_path / "run.pt").checkpoint().name == "run.pt"
    assert RunMetrics(tmp_path / "run.metrics.jsonl").checkpoint().name == "run.pt"


def test_recipe(tmp_path: Path):
    """A file of scores that cannot say which run produced them is what makes a control unverifiable."""
    metrics = RunMetrics(tmp_path / "run.pt")
    assert metrics.recipe() is None, "a run that has not described itself says so rather than guessing"

    metrics.describe({"crop_mm": "80.0", "centre": "tissue"})

    assert metrics.recipe() == {"crop_mm": "80.0", "centre": "tissue"}
