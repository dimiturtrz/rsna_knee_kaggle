from pathlib import Path

import pandas as pd

from kneemri.folds import ReportFolds


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
