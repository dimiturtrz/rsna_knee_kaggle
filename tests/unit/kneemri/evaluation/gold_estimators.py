"""The estimator family: what each one reads, and the recalibration none of them may see."""

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.gold_estimators import (
    Cells,
    MacroGold,
    PairCountGold,
    RankPooledGold,
)

GOLD = 40
"""Studies in a fixture gold set — enough that a bootstrap has something to resample."""

SEED = 7
"""Fixed, because a unit test may not draw from an unseeded generator."""

DRAWS = 16
"""Resamples in a test bootstrap. The registered count is 1000; this only has to exercise the path."""

SHIFT = 3.0
"""A per-finding score offset large enough to reorder cells ACROSS findings and no cells within one."""


def _cells(studies: int = GOLD) -> Cells:
    """Scores that rank the positives first in every finding, and the truth they are read against."""
    generator = np.random.default_rng(SEED)
    labels = (generator.random((studies, len(TARGETS))) < 0.4).astype(np.int64)
    labels[:, 0] = 0
    labels[0, 0] = 1
    scores = labels.astype(np.float64) + generator.random((studies, len(TARGETS))) * 0.1
    return Cells.of(scores, labels)


def _draws(studies: int = GOLD) -> list[np.ndarray]:
    generator = np.random.default_rng(SEED)
    return [generator.integers(0, studies, studies) for _ in range(DRAWS)]


def test_cells_of() -> None:
    """Rank-normalisation puts every finding on [0, 1] and leaves within-finding order untouched."""
    cells = _cells()
    assert cells.ranked.min() == 0.0
    assert cells.ranked.max() == pytest.approx(1.0)
    assert np.array_equal(np.argsort(cells.ranked, axis=0), np.argsort(cells.scores, axis=0))


def test_cells_of_disqualification_is_designed_out() -> None:
    """A per-finding shift reorders cells across findings and changes NO ranking inside one.

    Raw pooling moves 0.0288 under exactly this, which is why this module pools RANKS and never raw
    scores. Every implemented estimator must read the two identically — that is the property the
    registered ruler was chosen for, enforced here rather than asserted in prose.
    """
    cells = _cells()
    shifted = Cells.of((cells.scores + np.arange(len(TARGETS)) * SHIFT).astype(np.float64), cells.labels)
    for estimator in (RankPooledGold(), PairCountGold(), MacroGold()):
        assert estimator.of(shifted) == pytest.approx(estimator.of(cells))


def test_rows() -> None:
    cells = _cells()
    drawn = cells.rows(np.array([0, 0, 1], dtype=np.int64))
    assert drawn.scores.shape == (3, len(TARGETS))
    assert np.array_equal(drawn.ranked[0], cells.ranked[0])
    assert np.array_equal(drawn.labels[2], cells.labels[1])


def test_macro_gold_name() -> None:
    assert MacroGold().name == "macro"


def test_macro_gold_of() -> None:
    assert MacroGold().of(_cells()) == pytest.approx(1.0)


def test_pair_count_gold_name() -> None:
    assert PairCountGold().name == "paircount"


def test_pair_count_gold_of() -> None:
    assert PairCountGold().of(_cells()) == pytest.approx(1.0)


def test_rank_pooled_gold_name() -> None:
    assert RankPooledGold().name == "rankpooled"


def test_rank_pooled_gold_of() -> None:
    """Pooling ranks reads lower than a flat macro on the same cells — a different statistic, not a bug."""
    cells = _cells()
    pooled = RankPooledGold().of(cells)
    assert pooled > 0.9
    assert pooled < MacroGold().of(cells)


def test_of_single_class() -> None:
    """Nothing resolves when every gold cell carries the same label, so each estimator reports NaN."""
    labels = np.zeros((GOLD, len(TARGETS)), dtype=np.int64)
    cells = Cells.of(np.zeros((GOLD, len(TARGETS))), labels)
    for estimator in (RankPooledGold(), PairCountGold(), MacroGold()):
        assert np.isnan(estimator.of(cells))


def test_columns() -> None:
    """A weight is the pair count read off the TRUTH alone, never off either arm's areas."""
    cells = _cells()
    areas, weights = MacroGold().columns(cells)
    assert len(areas) == len(weights) == len(TARGETS)
    assert weights[0] == pytest.approx(GOLD - 1)


def test_spread() -> None:
    level, deviation = MacroGold().spread(_cells(), _draws())
    assert level == pytest.approx(MacroGold().of(_cells()))
    assert deviation >= 0.0


def test_difference() -> None:
    """A family read backwards is worse than itself read forwards, and the paired sd is finite."""
    cells = _cells()
    backwards = Cells.of(1.0 - cells.scores, cells.labels)
    difference, deviation = MacroGold().difference(cells, backwards, _draws())
    assert difference > 0.0
    assert deviation >= 0.0
