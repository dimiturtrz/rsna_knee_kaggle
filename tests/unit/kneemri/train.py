from pathlib import Path

import pytest

from kneemri.objective import Objective, RankedObjective
from kneemri.train import TrainingOptions


def test_parse(monkeypatch: pytest.MonkeyPatch):
    """Flags become a typed options object, so a run's configuration is inspectable without argv."""
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--labels", "labels.csv", "--fold", "3", "--epochs", "30", "--one-cycle", "--intensity-aug"],
    )
    options = TrainingOptions.parse()
    assert options.fold == 3
    assert options.epochs == 30
    assert options.one_cycle
    assert options.intensity_aug
    assert options.labels == Path("labels.csv")
    assert options.eval_labels is None, "no ruler given means score on what you trained on"


def test_parse_reads_a_separate_evaluation_ruler(monkeypatch: pytest.MonkeyPatch):
    """The ruler is separable, which is what makes two label arms comparable."""
    monkeypatch.setattr("sys.argv", ["train", "--labels", "a.csv", "--eval-labels", "b.csv"])
    options = TrainingOptions.parse()
    assert options.labels == Path("a.csv")
    assert options.eval_labels == Path("b.csv")


def test_parse_keeps_the_split_seed_independent_of_the_model_seed(monkeypatch: pytest.MonkeyPatch):
    """Changing the model seed must NOT move the fold split.

    This exists because it once did. `ReportFolds` was constructed with the training seed, so seed 7's
    fold-0 holdout shared 192 of 909 studies with seed 2026's — the two runs' holdout numbers were
    never comparable, and cross-scoring them evaluated a model on 717 studies it had trained on.
    """
    monkeypatch.setattr("sys.argv", ["train", "--labels", "a.csv", "--seed", "7"])
    options = TrainingOptions.parse()
    assert options.seed == 7
    assert options.split_seed == 2026, "the split is a property of the corpus, not of the run"


def test_objective():
    """The flags choose the loss, and the default must stay plain weighted cross-entropy.

    Checked without a GPU, a corpus or a downloaded encoder on purpose: the ranking arm sits behind a
    queue, and a miswired objective would otherwise surface only after the card had been waited on.
    """
    assert type(TrainingOptions().objective()) is Objective
    assert not TrainingOptions().objective().normalise_by_count

    ranked = TrainingOptions(rank_weight=0.5).objective()
    assert isinstance(ranked, RankedObjective)
    assert ranked.weight == 0.5

    assert TrainingOptions(rank_weight=0.5, normalise_by_count=True).objective().normalise_by_count
