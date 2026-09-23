import json
from pathlib import Path

import pytest

from kneemri.runs.run_metrics import RunMetrics
from kneemri.runs.run_record import EpochRow, RunRecord, StepRow

RESERVED_GB = 26.0
ALLOCATED_GB = 23.33


def test_rows(tmp_path: Path):
    """The record reads back as data, which is the whole reason it is not prose in a log."""
    metrics = RunMetrics(tmp_path / "member_fold0.pt")
    assert RunRecord(metrics.path).rows() == [], "a run that has not started yet is empty, not an error"

    metrics.record(EpochRow(epoch=0, loss=1.2, auc=0.73, seconds=30, per_label={"ACL": 0.7}))
    assert [row["epoch"] for row in RunRecord(metrics.path).rows()] == [0]


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

    assert [row["auc"] for row in RunRecord(metrics.path).rows()] == [0.71, 0.79], "the run the file's recipe describes"
    best = RunRecord(metrics.path).best()
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

    assert [row["auc"] for row in RunRecord(metrics.path).rows()] == [0.73, 0.83], "described late is still described"


def test_rows_splits_where_the_epoch_restarts(tmp_path: Path):
    """A back-filled recipe puts every run before it in one group, so the epoch has to say where they part.

    This is `lora_r8_v3small_224_s2026`: four epochs of an aborted run peaking 0.6867, then the real
    twelve peaking 0.8217, then the recipe. Read as one series it climbs +0.1350 from the gate epoch —
    two and a half times the largest real gain in the corpus `trajectory_check` decides a truncation on.
    """
    metrics = RunMetrics(tmp_path / "stitched.pt")
    for epoch, auc in enumerate([0.61, 0.65, 0.69]):
        metrics.record(EpochRow(epoch=epoch, loss=1.0, auc=auc, seconds=233, per_label={}))
    for epoch, auc in enumerate([0.79, 0.82]):
        metrics.record(EpochRow(epoch=epoch, loss=1.0, auc=auc, seconds=233, per_label={}))
    metrics.describe({"fold": "0", "epochs": "2"})

    assert [row["auc"] for row in RunRecord(metrics.path).rows()] == [0.79, 0.82], (
        "an epoch that does not increase is a new run"
    )


def test_epoch_seconds(tmp_path: Path):
    """One stalled epoch must not move the cost of a run, because on disk one already does.

    The shape is `pair_memory_s2026`'s: eleven epochs near 233 s and a single 27,266 s stall. The mean
    reads 2,690 s and would report the arm as 8x its control's cost; the median reads its real one.
    """
    metrics = RunMetrics(tmp_path / "stalled.pt")
    assert RunRecord(metrics.path).epoch_seconds() == 0.0, "a run that recorded nothing has no cost, not a crash"

    empty: dict[str, float] = {}
    for epoch, seconds in enumerate([233, 234, 233, 27266, 235, 233, 234]):
        metrics.record(EpochRow(epoch=epoch, loss=1.0, auc=0.8, seconds=seconds, per_label=empty))

    assert RunRecord(metrics.path).epoch_seconds() == 234


def test_stalls(tmp_path: Path):
    """The median made the stall harmless; this makes it visible, which is the other half.

    Same shape on disk as `pair_memory_s2026`. The run must name the 27,266 s epoch and NOTHING else —
    a per-epoch spread of 233-235 s is what a normal run looks like, and a rule that called any of those
    a stall would report one on every run in the directory.
    """
    metrics = RunMetrics(tmp_path / "stalled.pt")
    assert RunRecord(metrics.path).stalls() == [], "a run that recorded nothing has no stall, not a crash"

    empty: dict[str, float] = {}
    for epoch, seconds in enumerate([233, 234, 233, 27266, 235, 233, 234]):
        metrics.record(
            EpochRow(epoch=epoch, loss=1.0, auc=0.8, seconds=seconds, per_label=empty, started=1_760_000_000 + epoch)
        )

    stalls = RunRecord(metrics.path).stalls()
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

    assert RunRecord(metrics.path).stalls() == []


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
    best = RunRecord(checkpoint).best()
    assert best is not None and best["epoch"] == 0

    unscored = tmp_path / "final.pt"
    unscored.with_suffix(".metrics.jsonl").write_text(
        json.dumps({"epoch": 0, "loss": 1.0, "auc": float("nan"), "seconds": 9, "per_label": {}}) + "\n",
        encoding="utf-8",
    )
    assert RunRecord(unscored).best() is None

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
    assert RunRecord(indexed).best() is None, "an epoch index is not a score, whatever column it sits in"

    # THE EQUIVALENCE CLASS THE RECIPE GUARD CANNOT COVER: the same corruption in a file whose recipe
    # does not say fold -1. The guard is a provenance lookup, so it answers nothing here and the value
    # test is the only thing between an epoch index and the ledger. Row 0 survives it — 0.0 is a
    # legitimate AUC — which is the point: the reading falls back to the worst real row rather than
    # quoting the best fake one.
    untagged = tmp_path / "untagged.pt"
    untagged.with_suffix(".metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"recipe": {"fold": "0", "epochs": "12"}}),
                *(
                    json.dumps({"epoch": e, "loss": 1.0, "auc": float(e), "seconds": 9, "per_label": {}})
                    for e in range(3)
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    untagged_best = RunRecord(untagged).best()
    assert untagged_best is not None and untagged_best["auc"] == 1.0, "2.0 is not a probability and cannot win"


def test_exists(tmp_path: Path):
    """The recorder answers whether anything is there, so callers need not reach for its path.

    The distinction is worth making: a missing file is usually the wrong path, and an empty one is a run
    that died before its first scored epoch. They need different fixes.
    """
    metrics = RunMetrics(tmp_path / "run.pt")
    assert not RunRecord(metrics.path).exists()
    metrics.record(EpochRow(epoch=0, loss=1.0, auc=0.5, seconds=1, per_label={"ACL": 0.5}))
    assert RunRecord(metrics.path).exists()


def test_checkpoint(tmp_path: Path):
    """`.metrics.jsonl` is TWO suffixes, so the obvious `with_suffix` yields `run.metrics.pt`.

    That path is one nothing writes and everything then reads as a run carrying no stamp — which is
    how a checkpoint holding a full recipe reads as a run that recorded none. Both spellings of a run
    resolve to the same pair, because both reach this class.
    """
    assert RunRecord(tmp_path / "run.pt").checkpoint().name == "run.pt"
    assert RunRecord(tmp_path / "run.metrics.jsonl").checkpoint().name == "run.pt"


def test_recipe(tmp_path: Path):
    """A file of scores that cannot say which run produced them is what makes a control unverifiable."""
    metrics = RunMetrics(tmp_path / "run.pt")
    assert RunRecord(metrics.path).recipe() is None, "a run that has not described itself says so rather than guessing"

    metrics.describe({"crop_mm": "80.0", "centre": "tissue"})

    assert RunRecord(metrics.path).recipe() == {"crop_mm": "80.0", "centre": "tissue"}


def test_name(tmp_path: Path):
    """The run's own name, whichever of its two spellings the recorder was handed."""
    from_checkpoint = RunRecord(tmp_path / "band_dense_s2026.pt")
    from_record = RunRecord(tmp_path / "band_dense_s2026.metrics.jsonl")

    assert from_checkpoint.name() == "band_dense_s2026"
    assert from_record.name() == from_checkpoint.name(), "both spellings name the same run"


def test_optional_fields_round_trip(tmp_path: Path):
    """New NotRequired fields must still be read from files that carry them."""
    metrics = RunMetrics(tmp_path / "member.pt")
    metrics.record(
        EpochRow(
            epoch=0,
            loss=1.0,
            auc=0.8,
            seconds=10,
            per_label={"ACL": 0.7},
            ece=0.05,
            loss_q25=0.1,
            loss_q50=0.2,
            loss_q75=0.3,
            loss_q90=0.4,
            loss_q95=0.5,
        )
    )

    row = RunRecord(metrics.path).rows()[0]
    assert row["ece"] == pytest.approx(0.05)
    assert row["loss_q25"] == pytest.approx(0.1)
    assert row["loss_q95"] == pytest.approx(0.5)


def test_run(tmp_path: Path):
    """The identity line reads back apart from the recipe, and a run older than identities has none."""
    metrics = RunMetrics(tmp_path / "run.pt")
    metrics.begin({"slices": "12"})
    assert RunRecord(tmp_path / "run.pt").run() is None

    metrics.begin({"slices": "12"}, run={"id": "20260923T070000Z-abcd1234", "parents": []})
    metrics.record(EpochRow(epoch=0, loss=1.0, auc=0.8, seconds=1, per_label={"ACL": 0.8}))

    record = RunRecord(tmp_path / "run.metrics.jsonl")
    assert record.run() == {"id": "20260923T070000Z-abcd1234", "parents": []}
    assert [row["epoch"] for row in record.rows()] == [0], "an identity line is neither an epoch nor a boundary"


def test_steps(tmp_path: Path):
    """The step stream reads back in order, and a run older than it reads as having none."""
    record = RunRecord(tmp_path / "run.pt")
    assert record.steps() == []

    RunMetrics(tmp_path / "run.pt").steps_taken(
        [
            StepRow(epoch=0, step=0, loss=0.7, grad_norm=3.0, lr=[1e-4]),
            StepRow(epoch=0, step=1, loss=0.6, grad_norm=2.5, lr=[1e-4]),
        ]
    )

    assert [row["step"] for row in record.steps()] == [0, 1]
    assert record.steps()[0]["grad_norm"] == 3.0
