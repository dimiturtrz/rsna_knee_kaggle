from pathlib import Path

import numpy as np
import pandas as pd

from core.data.corpus.folds import TRAIN_ONLY, FitPositions, ReportFolds


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


def test_assign_keeps_a_foreign_corpus_out_of_every_holdout(tmp_path: Path):
    """A study this table never heard of trains and is never scored on — under a fold, and under -1.

    Otherwise a fifth of a merged cache's foreign studies land in the holdout, and the number the arm
    reports is over a different mix of corpora than the control it exists to be compared against.
    """
    folds = ReportFolds(folds=5)

    assignments = folds.assign(str(_table(tmp_path)), ["a", "b", "rijeka-vol01"])

    assert assignments[2] == TRAIN_ONLY
    for fold in (-1, 0, 1, 2, 3, 4):
        assert 2 not in folds.holdout(assignments, fold=fold)[1].tolist()


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


def test_sampled():
    """A cap that keeps the fold's ratio, and NOTHING at all when no cap is asked for."""
    training, holdout = np.arange(800), np.arange(800, 1000)
    uncapped = FitPositions().sampled(training, holdout)

    assert (uncapped[0] == training).all() and (uncapped[1] == holdout).all(), "no cap touches nothing"

    few_training, few_holdout = FitPositions(studies=50).sampled(training, holdout)

    assert (len(few_training), len(few_holdout)) == (40, 10), "the 4:1 the fold already has"
    assert (few_training == training[:40]).all(), "the FIRST positions, so two smokes read the same studies"

    one_training, one_holdout = FitPositions(studies=1).sampled(training, holdout)

    assert (list(one_training), list(one_holdout)) == ([0], [800]), "both halves keep a study"

    whole = FitPositions(studies=5000).sampled(training, holdout)

    assert (whole[0] == training).all() and (whole[1] == holdout).all(), "a cap cannot invent"


def test_thinned():
    """The data-scaling knob: one side cut at random, reproducibly, in reading order."""
    training = np.arange(800)

    assert (FitPositions().thinned(training) == training).all(), "the default is not a scaling run"

    half = FitPositions(train_fraction=0.5).thinned(training)

    assert len(half) == 400, "the share asked for"
    assert (np.sort(half) == half).all(), "ascending, so the loader still reads the cache sequentially"
    assert not (half == training[:400]).all(), (
        "drawn at random rather than sliced off the front — a prefix is a systematically chosen half, "
        "and any difference it produced could be a property of that half rather than of its size"
    )

    again = FitPositions(train_fraction=0.5).thinned(training)

    assert (again == half).all(), "seeded by --split-seed, so two arms at one fraction fit the same studies"
    assert len(FitPositions(train_fraction=0.0).thinned(training)) == 1, "a fraction cannot empty the fit"


def test_caveats():
    """A cut split says so, because the score it produces cannot be read against the ledger."""
    training = np.arange(400)

    assert FitPositions().caveats(training) == [], "an uncut run has nothing to disclaim"

    scaling = FitPositions(train_fraction=0.5).caveats(training)

    assert len(scaling) == 1
    assert scaling[0].startswith("DATA-SCALING RUN: 0.50")
    assert "400" in scaling[0], "the studies actually fitted, not the fraction alone"

    smoke = FitPositions(studies=60).caveats(training)

    assert len(smoke) == 1
    assert "not a result" in smoke[0]
    assert len(FitPositions(studies=60, train_fraction=0.5).caveats(training)) == 2, "both cuts, both said"
