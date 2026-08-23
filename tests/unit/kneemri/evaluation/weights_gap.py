"""Equivalence classes for the matched ours-vs-theirs ruler.

The partition is over what a published `oof` block can be relative to the run it is compared against:
a genuine held-out fold, a `fold: -1` block covering the whole corpus, and a seed whose folds do not
reach every study the run held out. Each is one class, plus the boundary at `HELD_OUT_SHARE`.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.random import Generator
from numpy.typing import NDArray

from core.findings import TARGETS
from kneemri.evaluation.weights_gap import HELD_OUT_SHARE, MatchedRuler, WeightsGap
from kneemri.runs.holdout_record import HoldoutRecord

STUDIES = [f"s{index:03d}" for index in range(20)]
LABELS = len(TARGETS)


def _member(
    directory: Path,
    name: str,
    seed: int,
    identifiers: list[str],
    generator: Generator,
) -> NDArray[np.float64]:
    """One published checkpoint carrying nothing but the two fields the reader takes."""
    values = generator.random((len(identifiers), LABELS))
    torch.save(
        {"config": {"seed": seed}, "oof": {"ids": list(identifiers), "pred": values.tolist()}},
        directory / f"{name}.pt",
    )
    return values


def _published(directory: Path, folds: int = 5, seeds: tuple[int, ...] = (1, 2)) -> MatchedRuler:
    """A published pool whose five folds partition STUDIES exactly, for each of `seeds` — 0.2 each."""
    generator = np.random.default_rng(7)
    step = len(STUDIES) // folds
    for seed in seeds:
        for fold in range(folds):
            _member(directory, f"m_{seed}_{fold}", seed, STUDIES[fold * step : (fold + 1) * step], generator)
    return MatchedRuler(directory)


def _record(directory: Path, studies: list[str] = STUDIES) -> Path:
    """A run record whose truth has both classes present in every label, so no AUC is undefined."""
    generator = np.random.default_rng(11)
    truth = np.zeros((len(studies), LABELS), dtype=np.float32)
    truth[::2] = 0.94
    path = directory / "arm.pt"
    HoldoutRecord(path).write(list(studies), generator.random((len(studies), LABELS)), truth, epoch=3)
    return path


def test_against_member() -> None:
    """The matched delta is ours minus their single member, not their pool."""
    row = WeightsGap(label="MCL", ours=0.80, their_member=0.75, their_pool=0.78, positives=9)
    assert row.against_member() == pytest.approx(0.05)


def test_against_pool() -> None:
    """The pooled delta reads against the four-seed average, which is a different question."""
    row = WeightsGap(label="MCL", ours=0.80, their_member=0.75, their_pool=0.78, positives=9)
    assert row.against_pool() == pytest.approx(0.02)


def test_folds(tmp_path: Path) -> None:
    """A block covering the whole corpus is refused; genuine folds are kept."""
    ruler = _published(tmp_path)
    _member(tmp_path, "m_trained_on_all", 1, STUDIES, np.random.default_rng(3))
    kept = ruler.folds()
    assert len(kept) == 10
    assert all(len(entry.identifiers) <= HELD_OUT_SHARE * len(STUDIES) for entry in kept)


def test_theirs(tmp_path: Path) -> None:
    """One column per seed, each row aligned to the studies asked for."""
    columns = _published(tmp_path).theirs(STUDIES)
    assert len(columns) == 2
    assert all(column.shape == (len(STUDIES), LABELS) for column in columns)


def test_theirs_drops_a_seed_that_does_not_cover(tmp_path: Path) -> None:
    """A seed whose folds miss any requested study is dropped rather than scored on a smaller population."""
    generator = np.random.default_rng(5)
    _member(tmp_path, "m_1_0", 1, STUDIES, generator)  # refused: whole corpus
    for fold in range(5):
        _member(tmp_path, f"m_2_{fold}", 2, STUDIES[fold * 4 : fold * 4 + 4], generator)
    _member(tmp_path, "m_3_0", 3, STUDIES[:4], generator)  # seed 3 never covers the rest
    assert len(MatchedRuler(tmp_path).theirs(STUDIES)) == 1


def test_compared(tmp_path: Path) -> None:
    """One row per finding, each carrying both readings and the holdout's positive count."""
    ruler = _published(tmp_path)
    rows = ruler.compared(_record(tmp_path))
    assert [row.label for row in rows] == list(TARGETS)
    assert all(row.positives == len(STUDIES) // 2 for row in rows)
    assert all(0.0 <= row.ours <= 1.0 for row in rows)


def test_compared_refuses_a_pool_that_covers_nothing(tmp_path: Path) -> None:
    """No seed reaching every study is a refusal, not a comparison over whichever studies survived."""
    generator = np.random.default_rng(13)
    _member(tmp_path, "m_1_0", 1, STUDIES[:4], generator)
    with pytest.raises(ValueError, match="no published seed covers"):
        MatchedRuler(tmp_path).compared(_record(tmp_path))
