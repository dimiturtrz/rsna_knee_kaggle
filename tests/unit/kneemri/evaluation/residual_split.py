"""The mirror for the residual decomposition: every public method called, and asserted."""

import numpy as np

from core.findings import TARGETS
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.residual_split import ResidualRow, ResidualSplit


def _row(unanimous: int = 3, minority: int = 1) -> ResidualRow:
    return ResidualRow(label="ACL", positives=10, missed=5, unanimous=unanimous, minority=minority)


def _pool(members: int = 3) -> Pool:
    truth = np.zeros((6, len(TARGETS)))
    truth[:3] = 1.0
    beliefs = np.tile(np.linspace(1.0, 0.0, 6)[:, np.newaxis], (1, len(TARGETS)))
    wrong = beliefs[::-1].copy()
    predictions = [beliefs if index else wrong for index in range(members)]
    return Pool(names=[f"m{index}" for index in range(members)], predictions=predictions, truth=truth)


def test_errors() -> None:
    assert _row().errors == 10


def test_unanimous_share() -> None:
    assert _row(unanimous=5).unanimous_share == 0.5


def test_minority_share() -> None:
    assert _row(minority=2).minority_share == 0.2


def test_line() -> None:
    line = _row().line()
    assert "ACL" in line
    assert "30.0%" in line


def test_called() -> None:
    beliefs = np.array([[0.9], [0.4], [0.1]])
    called = ResidualSplit.called(beliefs, np.array([2]))
    assert called[:, 0].tolist() == [True, True, False]


def test_truth() -> None:
    truth = ResidualSplit(_pool()).truth()
    assert truth.shape == (6, len(TARGETS))
    assert truth[:, 0].tolist() == [1, 1, 1, 0, 0, 0]


def test_mistakes() -> None:
    split = ResidualSplit(_pool())
    perfect = split.mistakes(np.tile(np.linspace(1.0, 0.0, 6)[:, np.newaxis], (1, len(TARGETS))))
    assert not perfect.any()


def test_per_member() -> None:
    members = ResidualSplit(_pool()).per_member()
    assert members.shape == (3, 6, len(TARGETS))
    assert members[0].all()
    assert not members[1].any()


def test_per_label() -> None:
    rows = ResidualSplit(_pool()).per_label()
    assert len(rows) == len(TARGETS)
    assert rows[0].positives == 3
    assert rows[0].unanimous == 0


def test_report() -> None:
    lines = ResidualSplit(_pool()).report()
    assert lines[0].startswith("3 members over 6 held-out studies")
    assert any("POOLED" in line for line in lines)
