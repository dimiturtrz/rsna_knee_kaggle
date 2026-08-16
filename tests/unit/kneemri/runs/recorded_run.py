"""A run's account of itself when only one of its two records survives, and when they disagree."""

from pathlib import Path

import torch

from kneemri.runs.recorded_run import RecordedRun
from kneemri.runs.run_metrics import EpochRow, RunMetrics


def _scored(checkpoint: Path, aucs: list[float], recipe: dict[str, str] | None = None) -> RunMetrics:
    metrics = RunMetrics(checkpoint)
    if recipe is not None:
        metrics.describe(recipe)
    for epoch, auc in enumerate(aucs):
        metrics.record(EpochRow(epoch=epoch, loss=1.0, auc=auc, seconds=9, per_label={"ACL": auc}))
    return metrics


def test_of(tmp_path: Path):
    """The metrics file wins where it exists: it names the epoch that SCORED best, not the last saved."""
    checkpoint = tmp_path / "arm.pt"
    torch.save({"options": {"scheme": "canonical"}, "epoch": 9}, checkpoint)
    metrics = _scored(checkpoint, [0.80, 0.86, 0.84], {"scheme": "extended"})

    recorded = RecordedRun.of(metrics)

    assert recorded.recipe == {"scheme": "extended"}, "a run that wrote its own account has answered"
    assert recorded.kept_epoch == 1, "under patience the later epochs' weights were thrown away"


def test_of_reads_a_run_that_kept_only_its_checkpoint(tmp_path: Path):
    """50 of 69 members wrote no metrics file, and 36 of those stamp the whole recipe anyway.

    `wsfold0_e6` is the one that cost something: the control knee-bot's finding rests on, read as
    "recipe unrecorded, so nothing here can say the two runs are matched" while every option it ran
    under sat in the checkpoint the comparison had already been handed.
    """
    checkpoint = tmp_path / "stamped.pt"
    torch.save({"options": {"scheme": "canonical", "epochs": "12"}, "epoch": 6}, checkpoint)

    recorded = RecordedRun.of(RunMetrics(checkpoint))

    assert recorded.recipe == {"scheme": "canonical", "epochs": "12"}
    assert recorded.kept_epoch == 6, "the saved epoch, which only the stamp knows"


def test_of_reads_a_run_whose_scores_predate_the_recipe(tmp_path: Path):
    """The two facts come from different records for the same run, so neither source is read as a pair."""
    checkpoint = tmp_path / "split.pt"
    torch.save({"options": {"scheme": "canonical"}}, checkpoint)
    metrics = _scored(checkpoint, [0.80, 0.86])

    recorded = RecordedRun.of(metrics)

    assert recorded.recipe == {"scheme": "canonical"}, "from the stamp, which is the only one that has it"
    assert recorded.kept_epoch == 1, "from the scores, which name the best rather than the last"


def test_of_states_an_absence_rather_than_an_empty_recipe(tmp_path: Path):
    """14 members predate options entirely, and an empty recipe would compare as matched against anything."""
    checkpoint = tmp_path / "ancient.pt"
    torch.save({"model": {}, "holdout": 0.81}, checkpoint)

    recorded = RecordedRun.of(RunMetrics(checkpoint))

    assert recorded.recipe is None
    assert recorded.kept_epoch is None, "not derived from a count — that is what stamped an epoch-6 best as 11"
