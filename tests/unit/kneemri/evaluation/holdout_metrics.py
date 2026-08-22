"""A run that predates the recorder still kept the arrays its numbers were read off."""

import json
from pathlib import Path

import numpy as np

from core.findings import TARGETS
from kneemri.evaluation.holdout_metrics import HoldoutMetrics, ScoredEpoch
from kneemri.runs.holdout_record import HoldoutRecord
from kneemri.runs.run_metrics import EpochRow, RunMetrics

STUDIES = 40


def _kept(checkpoint: Path, epoch: int = 6) -> None:
    """A holdout record whose predictions are genuinely separable, so the AUCs are not all 0.5."""
    truth = (np.random.default_rng(2026).random((STUDIES, len(TARGETS))) < 0.5).astype(np.float32)
    noise = np.random.default_rng(11).random((STUDIES, len(TARGETS)))
    HoldoutRecord(checkpoint).write(
        [f"s{index}" for index in range(STUDIES)],
        (truth * 0.6 + noise * 0.4).astype(np.float64),
        truth,
        epoch=epoch,
    )


def test_epoch(tmp_path: Path):
    """The AUCs come out of the kept arrays; what was never in them comes back as absent, not guessed."""
    checkpoint = tmp_path / "wsfold0_e6.pt"
    _kept(checkpoint)
    reading = HoldoutRecord(checkpoint).read()
    assert reading is not None

    row = HoldoutMetrics().epoch(reading)

    assert row["epoch"] == 6, "the epoch the kept weights belong to, not a position in a list"
    assert set(row["per_label"]) == set(TARGETS)
    assert row["auc"] > 0.5, "predictions built to rank the truth must score above chance"
    assert np.isnan(row["loss"]), "the loss was never in the array, so it is absent rather than plausible"
    assert row["seconds"] == 0, "and so is the wall-clock"


def test_named():
    """A reconstructed control must never read as a recorded one — the report says which it holds."""
    row: EpochRow = {"epoch": 1, "loss": 1.0, "auc": 0.8, "seconds": 9, "per_label": {}}

    assert "RECONSTRUCTED" in ScoredEpoch(row, reconstructed=True).named()
    assert "RECONSTRUCTED" not in ScoredEpoch(row, reconstructed=False).named()


def test_of(tmp_path: Path):
    """Recorded scores win; predictions answer for the runs that predate the recorder; a final fit does not."""
    resolver = HoldoutMetrics()

    absent = RunMetrics(tmp_path / "nothing.pt")
    assert resolver.of(absent) is None, "neither record can say, so nothing does"

    predates = tmp_path / "wsfold0_e6.pt"
    _kept(predates)
    recovered = resolver.of(RunMetrics(predates))
    assert recovered is not None and recovered.reconstructed
    assert recovered.row["epoch"] == 6

    recorded = RunMetrics(tmp_path / "arm.pt")
    recorded.record(EpochRow(epoch=0, loss=1.0, auc=0.84, seconds=300, per_label={"ACL": 0.7}))
    _kept(tmp_path / "arm.pt")
    kept = resolver.of(recorded)
    assert kept is not None and not kept.reconstructed, "a run that scored itself is not recomputed over"
    assert kept.row["seconds"] == 300, "and it keeps the wall-clock a reconstruction cannot recover"

    # A final fit holds nothing back, so a number computed from any file beside it would be a score on
    # training data wearing a holdout's name. `RunMetrics.best` refuses one on its recipe and this must
    # obey the same rule, or reconstruction becomes the way around it.
    final = tmp_path / "final_fit.pt"
    final.with_suffix(".metrics.jsonl").write_text(
        json.dumps({"recipe": {"fold": "-1", "epochs": "12"}}) + "\n", encoding="utf-8"
    )
    _kept(final)
    assert resolver.of(RunMetrics(final)) is None, "a run that held nothing back has no holdout to score"
