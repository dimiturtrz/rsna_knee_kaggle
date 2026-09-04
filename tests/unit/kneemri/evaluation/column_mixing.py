"""Unit cover for the cross-column mixing probe.

The equivalence classes are MECHANICAL — a rank column spans [0, 1], a ridge row is one weight per
finding, a matrix is square, a scored table reports one row per target and never scores a study its own
fit saw. Whether mixing PAYS is a measurement over recorded holdouts and belongs to the module's run.
"""

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.column_mixing import ColumnMixing, MixingResult
from kneemri.evaluation.mixing_rules import RULES, RidgeShrink, SparseGraph

STUDIES = 120
FINDINGS = len(TARGETS)


def corpus() -> tuple[np.ndarray, np.ndarray]:
    """A table whose first column carries the second column's signal, so mixing has something to find."""
    generator = np.random.default_rng(0)
    truth = generator.integers(0, 2, size=(STUDIES, FINDINGS))
    table = generator.normal(size=(STUDIES, FINDINGS)) + truth
    table[:, 0] = generator.normal(size=STUDIES)
    table[:, 1] = generator.normal(size=STUDIES) + truth[:, 0]
    return table, truth


def test_column_mixing_ranks() -> None:
    """Every column becomes its own rank on [0, 1], order preserved and endpoints hit."""
    table = np.random.default_rng(1).normal(size=(STUDIES, FINDINGS))
    normalised = ColumnMixing.ranks(table)
    assert normalised.shape == table.shape
    assert normalised.min() == 0.0
    assert normalised.max() == 1.0
    assert np.array_equal(np.argsort(normalised[:, 0]), np.argsort(table[:, 0]))


def test_chosen_penalty() -> None:
    """The penalty comes from the rule's own candidates, chosen inside the fitting half."""
    table, truth = corpus()
    penalty = ColumnMixing().chosen_penalty(ColumnMixing.ranks(table), truth, 0)
    assert penalty in set(RidgeShrink.penalties)
    assert ColumnMixing(fixed=7.0).chosen_penalty(ColumnMixing.ranks(table), truth, 0) == 7.0


def test_sparse_rule_is_carried() -> None:
    """A rule handed in is the one fitted — the sparse matrix keeps its own zero diagonal weight of one."""
    table, truth = corpus()
    matrix = ColumnMixing(rule=SparseGraph(), fixed=1.0).matrix(ColumnMixing.ranks(table), truth)
    assert set(RULES) == {"ridge", "sparse"}
    assert np.allclose(np.diag(matrix), 1.0)


def test_matrix() -> None:
    """The mixing matrix is square in the findings and finite everywhere."""
    table, truth = corpus()
    matrix = ColumnMixing().matrix(ColumnMixing.ranks(table), truth)
    assert matrix.shape == (FINDINGS, FINDINGS)
    assert np.isfinite(matrix).all()


def test_scored() -> None:
    """One row per finding, in target order, each carrying its own positive count and two areas."""
    table, truth = corpus()
    results = ColumnMixing().scored(table, truth)
    assert [result.finding for result in results] == list(TARGETS)
    assert all(result.scored == STUDIES for result in results)
    assert all(result.positives == int(truth[:, column].sum()) for column, result in enumerate(results))
    assert all(0.0 <= result.mixed <= 1.0 for result in results)


def test_delta() -> None:
    """The delta is the mixed area minus the shipped one."""
    result = MixingResult(finding="Fracture", scored=909, positives=62, plain=0.90, mixed=0.93)
    assert result.delta == pytest.approx(0.03)


def test_line() -> None:
    """The printed row names the finding, both areas, a signed delta and the positives."""
    line = MixingResult(finding="Fracture", scored=909, positives=62, plain=0.90, mixed=0.93).line()
    assert "Fracture" in line
    assert "+0.0300" in line
    assert "pos=62" in line
