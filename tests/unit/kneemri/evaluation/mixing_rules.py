"""Unit cover for the two mixing rules.

The equivalence class that matters is the DEGENERATE LIMIT: each rule must put the shipped table — the
identity row — at the far end of its own penalty path, because a comparison whose control cannot be
selected is not a comparison. That is the failure this module exists to have already fixed once.
"""

import numpy as np

from core.findings import TARGETS
from kneemri.evaluation.mixing_rules import RidgeShrink, SparseGraph

STUDIES = 120
FINDINGS = len(TARGETS)


def corpus() -> tuple[np.ndarray, np.ndarray]:
    """A design whose second column carries the first column's truth, so a cross term has something to find."""
    generator = np.random.default_rng(0)
    truth = generator.integers(0, 2, size=(STUDIES, FINDINGS))
    design = generator.normal(size=(STUDIES, FINDINGS)) + truth
    design[:, 1] = generator.normal(size=STUDIES) + truth[:, 0]
    return design, truth


def identity_row() -> np.ndarray:
    """The shipped table's row for finding zero: keep the own column, add nothing."""
    row = np.zeros(FINDINGS)
    row[0] = 1.0
    return row


def walks_to_identity(rule: RidgeShrink | SparseGraph, tight: float) -> None:
    """One weight per finding, and the tight penalty sits closer to the shipped row than the loose one."""
    design, truth = corpus()
    loose = rule.row(design, truth[:, 0], min(rule.penalties), 0)
    pinned = rule.row(design, truth[:, 0], tight, 0)
    assert loose.shape == (FINDINGS,)
    assert np.abs(pinned - identity_row()).sum() < np.abs(loose - identity_row()).sum()
    assert np.allclose(pinned, identity_row(), atol=1e-4)


def test_ridge_shrink_row() -> None:
    """The dense rule shrinks toward the identity, not toward zero — the bug this module was rewritten for."""
    walks_to_identity(RidgeShrink(), 1e8)


def test_sparse_graph_row() -> None:
    """The sparse rule reaches the identity too, never scales its own column, and does zero cross terms."""
    walks_to_identity(SparseGraph(), 10.0)
    design, truth = corpus()
    row = SparseGraph().row(design, truth[:, 0], 0.03, 0)
    assert row[0] == 1.0
    assert (row[1:] == 0.0).any()
