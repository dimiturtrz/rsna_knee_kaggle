import json
from pathlib import Path

from kneemri.runs.run_metrics import EpochRow, RunMetrics


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


def test_describe(tmp_path: Path):
    """The recipe shares the file with the scores, so every reader of the scores must ignore it."""
    metrics = RunMetrics(tmp_path / "run.pt")

    metrics.describe({"crop_mm": "80.0", "centre": "tissue"})
    metrics.record(EpochRow(epoch=0, loss=1.0, auc=0.5, seconds=1, per_label={"ACL": 0.5}))

    assert [row["epoch"] for row in metrics.rows()] == [0], "the recipe line is not an epoch"
    assert metrics.best() is not None, "and ranking epochs must not trip over it"


def test_recipe(tmp_path: Path):
    """A file of scores that cannot say which run produced them is what makes a control unverifiable."""
    metrics = RunMetrics(tmp_path / "run.pt")
    assert metrics.recipe() is None, "a run that has not described itself says so rather than guessing"

    metrics.describe({"crop_mm": "80.0", "centre": "tissue"})

    assert metrics.recipe() == {"crop_mm": "80.0", "centre": "tissue"}
