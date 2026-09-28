import numpy as np
import torch

from core.findings import TARGETS
from kneemri.training.early_targets import EarlyTargets

BETA = 0.7


def test_rows():
    """A fresh history is all zeros — the paper's warm-up — and is shaped (batch, finding, 2)."""
    rows = EarlyTargets(4, BETA).rows(np.array([0, 3], dtype=np.int64))

    assert rows.shape == (2, len(TARGETS), 2)
    assert not rows.any()


def test_learn():
    """One visit folds in (1 - beta) of the reading; an unvisited study stays untouched."""
    history = EarlyTargets(3, BETA)
    history.learn(np.array([1], dtype=np.int64), torch.zeros(1, len(TARGETS)))

    rows = history.rows(np.array([0, 1], dtype=np.int64))
    np.testing.assert_allclose(rows[1], np.full((len(TARGETS), 2), (1 - BETA) * 0.5), rtol=1e-6)
    assert not rows[0].any()
