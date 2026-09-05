"""Unit cover for the per-column headroom report.

The equivalence classes are ARITHMETIC — a shortfall floors at zero, a board value is the sum of
shortfalls over twelve, a pooled table is a rank average on [0, 1], and the report comes back sorted
worst column first. Whether a weak column can actually be lifted is a measurement over recorded
holdouts and belongs to the module's run, never here.
"""

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.column_headroom import ColumnHeadroom, HeadroomRow
from kneemri.evaluation.stack_fit import MemberHoldout

STUDIES = 120
FINDINGS = len(TARGETS)


def members(count: int = 3) -> list[MemberHoldout]:
    """A few members over one shared holdout, the first column carrying no signal so it ranks last."""
    generator = np.random.default_rng(0)
    truth = generator.integers(0, 2, size=(STUDIES, FINDINGS))
    studies = np.asarray([f"study{index:04d}" for index in range(STUDIES)])
    built = []
    for index in range(count):
        predictions = generator.normal(size=(STUDIES, FINDINGS)) + truth
        predictions[:, 0] = generator.normal(size=STUDIES)
        built.append(MemberHoldout(f"member{index}", studies, predictions, truth))
    return built


def test_pooled() -> None:
    """The pool is a rank average on [0, 1], one value per cell of the shared holdout."""
    table = ColumnHeadroom.pooled(members())
    assert table.shape == (STUDIES, FINDINGS)
    assert table.min() >= 0.0
    assert table.max() <= 1.0


def test_scored() -> None:
    """One row per finding, sorted worst first, each carrying its own positives — and the null column is last."""
    pool = members()
    rows = ColumnHeadroom().scored(ColumnHeadroom.pooled(pool), pool[0].truth)
    assert {row.finding for row in rows} == set(TARGETS)
    assert [row.area for row in rows] == sorted(row.area for row in rows)
    assert rows[0].finding == TARGETS[0]
    assert all(row.scored == STUDIES for row in rows)
    assert all(row.positives == int(pool[0].truth[:, TARGETS.index(row.finding)].sum()) for row in rows)


def test_board_value() -> None:
    """The board value is the summed shortfall over twelve, and a pool already above target is worth zero."""
    rows = [HeadroomRow(finding=finding, scored=909, positives=100, area=0.80) for finding in TARGETS]
    assert ColumnHeadroom(0.90).board_value(rows) == pytest.approx(0.10)
    assert ColumnHeadroom(0.70).board_value(rows) == pytest.approx(0.0)


def test_shortfall() -> None:
    """A column below the target owes the difference; one above owes nothing rather than a credit."""
    row = HeadroomRow(finding="MCL", scored=909, positives=171, area=0.8104)
    assert row.shortfall(0.90) == pytest.approx(0.0896)
    assert row.shortfall(0.80) == 0.0


def test_line() -> None:
    """The printed row names the finding, its area, its shortfall, the board points and the positives."""
    line = HeadroomRow(finding="MCL", scored=909, positives=171, area=0.8104).line(0.90)
    assert "MCL" in line
    assert "0.8104" in line
    assert "+0.0896" in line
    assert "pos=171" in line
