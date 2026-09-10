"""The stitched gold read: what it aligns, what it refuses, and what a partial glob would have cost."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.bootstrap import StudyBootstrap
from core.findings import TARGETS
from kneemri.evaluation.gold_estimators import Cells, MacroGold
from kneemri.evaluation.oof_gold import CORPUS_STUDIES, GoldRuler, Stitch

GOLD = 40
"""Annotated studies in the fixture corpus — enough that a bootstrap has something to resample."""

SEED = 7
"""Fixed, because a unit test may not draw from an unseeded generator."""

DRAWS = 16
"""Resamples in a test bootstrap. The registered count is 1000; this only has to exercise the path."""


def _identifier(index: int) -> str:
    return f"study_{index:05d}"


def _corpus(tmp_path: Path, family: str, *, folds: int = 2, backwards: bool = False) -> tuple[Path, Path]:
    """A whole-corpus fold family on disk, plus the train.csv whose first `GOLD` rows are annotated."""
    generator = np.random.default_rng(SEED)
    names = [_identifier(index) for index in range(CORPUS_STUDIES)]
    truth = (generator.random((CORPUS_STUDIES, len(TARGETS))) < 0.4).astype(np.float64)
    scores = truth + generator.random((CORPUS_STUDIES, len(TARGETS))) * 0.1
    if backwards:
        scores = 1.0 - scores
    block = CORPUS_STUDIES // folds
    for fold in range(folds):
        start, stop = fold * block, (fold + 1) * block if fold < folds - 1 else CORPUS_STUDIES
        np.savez(
            tmp_path / f"{family}_fold{fold}.holdout.npz",
            studies=np.array(names[start:stop]),
            predictions=scores[start:stop],
        )
    table = pd.DataFrame({"StudyInstanceUID": names})
    for index, target in enumerate(TARGETS):
        column = np.full(CORPUS_STUDIES, np.nan)
        column[:GOLD] = truth[:GOLD, index]
        table[target] = column
    train = tmp_path / f"{family}_train.csv"
    table.to_csv(train, index=False)
    return train, tmp_path


def _ruler() -> GoldRuler:
    return GoldRuler(bootstrap=StudyBootstrap(resamples=DRAWS))


def _stitched(tmp_path: Path, ruler: GoldRuler, family: str = "fam") -> Stitch:
    train, models = _corpus(tmp_path, family)
    return ruler.stitch(family, tuple(sorted(models.glob(f"{family}_fold*.npz"))), train)


def test_positives() -> None:
    labels = np.zeros((GOLD, len(TARGETS)), dtype=np.int64)
    labels[0, 0] = 1
    stitch = Stitch(family="f", cells=Cells.of(np.zeros((GOLD, len(TARGETS))), labels), covered=GOLD)
    positives = stitch.positives()
    assert set(positives) == set(TARGETS)
    assert positives[TARGETS[0]] == 1


def test_stitch(tmp_path: Path) -> None:
    stitch = _stitched(tmp_path, _ruler())
    assert stitch.covered == GOLD
    assert stitch.cells.labels.shape == (GOLD, len(TARGETS))
    assert MacroGold().of(stitch.cells) > 0.9


def test_stitch_refuses_partial_corpus(tmp_path: Path) -> None:
    """A glob short of the corpus drops gold studies silently and still returns a plausible macro."""
    train, models = _corpus(tmp_path, "fam")
    with pytest.raises(ValueError, match="a fold is missing"):
        _ruler().stitch("fam", (next(iter(sorted(models.glob("fam_fold*.npz")))),), train)


def test_draws(tmp_path: Path) -> None:
    ruler = _ruler()
    drawn = ruler.draws(_stitched(tmp_path, ruler))
    assert len(drawn) == DRAWS
    assert all(rows.shape == (GOLD,) for rows in drawn)


def test_spread(tmp_path: Path) -> None:
    ruler = _ruler()
    stitch = _stitched(tmp_path, ruler)
    level, deviation = ruler.spread(stitch, MacroGold())
    assert level == pytest.approx(MacroGold().of(stitch.cells))
    assert deviation >= 0.0


def test_paired(tmp_path: Path) -> None:
    """A family read backwards is worse than itself read forwards, by more than the paired sd."""
    ruler = _ruler()
    stitch = _stitched(tmp_path, ruler)
    train, models = _corpus(tmp_path, "back", backwards=True)
    against = ruler.stitch("back", tuple(sorted(models.glob("back_fold*.npz"))), train)
    difference, deviation = ruler.paired(stitch, against, MacroGold())
    assert difference > 0.0
    assert deviation >= 0.0


def test_report(tmp_path: Path) -> None:
    ruler = _ruler()
    stitch = _stitched(tmp_path, ruler)
    body = "\n".join(ruler.report(stitch))
    assert "rankpooled" in body
    assert "gold studies" in body
    assert "PAIRED" not in body
    assert "PAIRED" in "\n".join(ruler.report(stitch, stitch))
