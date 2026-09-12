"""Soft pseudo-labels: how a pool's spread becomes a weight, and how the table joins.

The equivalence classes are the ones the mechanism turns on: a unanimous pool, a split pool, a pool of
one, and an index read against a cache whose foreign block does not start at zero.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.inference.pseudo_labels import CONFIDENCE, IDENTIFIER, PseudoLabels

FINDINGS = len(TARGETS)


def _beliefs(*values: float) -> list[np.ndarray]:
    """One study, one constant belief per teacher — the shape `pooled` reduces over."""
    return [np.full((1, FINDINGS), value, dtype=np.float64) for value in values]


def test_pooled():
    """The mean is the level, and unanimity is what earns the full weight."""
    levels, weights = PseudoLabels.pooled(_beliefs(0.8, 0.8, 0.8))

    assert levels == pytest.approx(0.8)
    assert weights == pytest.approx(1.0), "teachers that agree teach at full weight"


def test_pooled_discounts_a_split_pool():
    """A maximally split pool lands at zero, and everything between is discounted in proportion."""
    _, maximal = PseudoLabels.pooled(_beliefs(0.0, 1.0))
    _, wide = PseudoLabels.pooled(_beliefs(0.1, 0.9))
    _, narrow = PseudoLabels.pooled(_beliefs(0.50, 0.54))

    assert maximal == pytest.approx(0.0), "sd 0.5 is the most a probability can have, and it costs everything"
    assert wide == pytest.approx(0.2), "agreeing on the sign and not the level is weak, not silenced"
    assert 0.9 < float(narrow.min()) < 1.0, "a small disagreement discounts a little, not everything"


def test_pooled_takes_a_pool_of_one():
    """One teacher has no spread to discount, so it teaches at full weight rather than at zero."""
    levels, weights = PseudoLabels.pooled(_beliefs(0.3))

    assert levels == pytest.approx(0.3)
    assert weights == pytest.approx(1.0)


def test_table():
    """Every finding gets a level column and a weight column, under the identifier the trainer joins on."""
    levels = np.linspace(0.0, 1.0, 2 * FINDINGS).reshape(2, FINDINGS)

    frame = PseudoLabels.table(["a", "b"], levels, np.ones_like(levels))

    assert list(frame[IDENTIFIER]) == ["a", "b"]
    assert frame[TARGETS[0]][0] == pytest.approx(levels[0, 0])
    assert frame[f"{TARGETS[-1]}{CONFIDENCE}"][1] == pytest.approx(1.0)
    assert len(frame.columns) == 1 + 2 * FINDINGS, "one identifier, a level and a weight per finding"


def test_rows(tmp_path: Path):
    """The identifiers come out in order and the positions start where the foreign block was appended."""
    index = tmp_path / "index.json"
    index.write_text(json.dumps(["2020_1", "2020_2"]))

    identifiers, positions = PseudoLabels.rows(index, 4407)

    assert identifiers == ["2020_1", "2020_2"]
    assert positions.tolist() == [4407, 4408], "labelling from zero would name our own studies as theirs"
