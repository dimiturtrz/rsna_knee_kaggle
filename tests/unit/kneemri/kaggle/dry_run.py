"""The dry run's scratch tables, which are what let the shipped path run over labelled studies."""

from pathlib import Path

import pandas as pd
import torch

from core.paths import DataRoot
from kneemri.findings import TARGETS
from kneemri.kaggle.dry_run import DryRun


def _root(base: Path) -> DataRoot:
    """A real data root whose raw half holds three studies and four series."""
    root = DataRoot(base)
    root.raw().mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"StudyInstanceUID": ["a", "b", "c"]}).to_csv(root.raw() / "train.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ["a", "a", "b", "c"], "SeriesInstanceUID": ["1", "2", "3", "4"]}).to_csv(
        root.raw() / "train_series.csv", index=False
    )
    return root


def test_tables(tmp_path: Path):
    """Only the asked-for studies reach the scratch tables, series included.

    A subset is written rather than teaching Submission a limit, because the point of the dry run is to
    exercise the SHIPPED code rather than a variant of it that only ever runs here.
    """
    root = _root(tmp_path)

    scratch = DryRun(root, torch.device("cpu")).tables(["a", "c"], tmp_path / "scratch")

    assert sorted(pd.read_csv(scratch / "train.csv")["StudyInstanceUID"]) == ["a", "c"]
    series = pd.read_csv(scratch / "train_series.csv")
    assert sorted(series["StudyInstanceUID"]) == ["a", "a", "c"], "a study's every series travels with it"


def test_scored(tmp_path: Path):
    """A table that ranks the positives first scores near one, and a constant table near a half.

    The constant case is the one worth pinning: it is what a crashed run's fallback wrote twice, and
    on the leaderboard it is indistinguishable from a model that learned nothing.
    """
    root = _root(tmp_path)
    labels = tmp_path / "labels.csv"
    frame = pd.DataFrame({"StudyInstanceUID": ["a", "b", "c"]})
    truth = {"a": 1.0, "b": 0.0, "c": 0.0}
    pd.DataFrame({"StudyInstanceUID": list(truth), **{t: [truth[s] for s in truth] for t in TARGETS}}).to_csv(
        labels, index=False
    )
    for target in TARGETS:
        frame[target] = [0.9, 0.1, 0.2]

    ranked = DryRun(root, torch.device("cpu")).scored(frame, labels)

    assert ranked["mean"] == 1.0, "a table that ranks the positive first is perfect"

    for target in TARGETS:
        frame[target] = [0.5, 0.5, 0.5]
    assert DryRun(root, torch.device("cpu")).scored(frame, labels)["mean"] == 0.5, "a constant table is a coin"
