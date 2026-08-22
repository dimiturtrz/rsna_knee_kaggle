from pathlib import Path

import numpy as np
import pandas as pd

from core.data.corpus.folds import ReportFolds


def _table(tmp_path: Path) -> Path:
    """Six studies, three of which share one report verbatim — the template-report leak in miniature."""
    path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "b", "c", "d", "e", "f"],
            "Report": ["normal knee", "normal knee", "normal knee", "acl tear", "meniscal tear", "effusion"],
        }
    ).to_csv(path, index=False)
    return path


def test_assign(tmp_path: Path):
    """Studies sharing a report get the same fold, so a duplicate group cannot straddle the divide."""
    identifiers = ["a", "b", "c", "d", "e", "f"]
    assignments = ReportFolds(folds=5).assign(str(_table(tmp_path)), identifiers)
    assert len(set(assignments[:3])) == 1, "the three template studies must share a fold"
    assert assignments.shape == (6,)


def test_assign_is_stable_across_runs(tmp_path: Path):
    """The split is a hash, not a shuffle: the same corpus gives the same folds on any machine."""
    identifiers = ["a", "b", "c", "d", "e", "f"]
    table = str(_table(tmp_path))
    assert ReportFolds().assign(table, identifiers).tolist() == ReportFolds().assign(table, identifiers).tolist()


def test_holdout(tmp_path: Path):
    """Training and holdout partition the corpus exactly, with no study in both."""
    identifiers = ["a", "b", "c", "d", "e", "f"]
    folds = ReportFolds(folds=3)
    assignments = folds.assign(str(_table(tmp_path)), identifiers)
    training, holdout = folds.holdout(assignments, fold=0)
    assert set(training.tolist()) & set(holdout.tolist()) == set()
    assert len(training) + len(holdout) == len(identifiers)


def test_holdout_with_fold_minus_one_trains_on_everything():
    """The final-fit mode: no fold is ever assigned -1, so nothing is held back.

    Asserted rather than left implicit because train.py's --fold -1 depends on exactly this, and a
    future change to how folds are numbered could take it away without any other test noticing.
    """
    assignments = np.array([0, 1, 2, 3, 4, 0, 1], dtype=np.int64)
    training, holdout = ReportFolds(5, 2026).holdout(assignments, -1)
    assert len(holdout) == 0
    assert training.tolist() == list(range(len(assignments)))
