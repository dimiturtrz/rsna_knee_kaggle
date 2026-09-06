"""Unit tests for the ruler swap.

Built on a pool whose answer is known by construction: one member agrees with table A and disagrees with
table B, another does the reverse, so the greedy MUST seat them in opposite orders under the two tables.
That is the whole claim the tool makes at scale — that a seat order is a property of the ruler.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from core.findings import TARGETS
from kneemri.evaluation.pool import Pool
from kneemri.kaggle.ruler_swap import RulerSwap, Seat

STUDIES = 40
STUDY_COLUMN = "StudyInstanceUID"


def _identifiers() -> list[str]:
    return [f"s{index:03d}" for index in range(STUDIES)]


def _table(path: Path, values: np.ndarray) -> Path:
    frame = pd.DataFrame(values, columns=list(TARGETS))
    frame.insert(0, STUDY_COLUMN, _identifiers())
    frame.to_csv(path, index=False)
    return path


def _train_table(path: Path) -> Path:
    frame = pd.DataFrame({STUDY_COLUMN: _identifiers()})
    frame.to_csv(path, index=False)
    return path


def _alternating() -> np.ndarray:
    column = np.tile([1.0, 0.0], STUDIES // 2)
    return np.repeat(column[:, None], len(TARGETS), axis=1)


def _swap(tmp_path: Path, seats: int = 2) -> RulerSwap:
    return RulerSwap(tmp_path, _train_table(tmp_path / "train.csv"), members=["a", "b"], seats=seats)


def _pool(truth: np.ndarray) -> Pool:
    """One member that reads the truth and one that reads its opposite."""
    return Pool(names=["reader", "contrarian"], predictions=[truth.copy(), 1.0 - truth], truth=truth)


def test_truth(tmp_path: Path) -> None:
    """A table is read onto the studies IN ORDER, and unread cells are masked rather than scored."""
    values = _alternating()
    values[0, 0] = np.nan
    scores, mask = _swap(tmp_path).truth(_table(tmp_path / "v.csv", values), _identifiers())

    assert scores.shape == (STUDIES, len(TARGETS))
    assert not mask[0, 0], "a cell the table did not read carries no weight"
    assert mask[1, 0], "and one it did read does"


def test_swapped(tmp_path: Path) -> None:
    """Only the truth changes: the members and their predictions come through untouched."""
    truth = _alternating()
    pool = _pool(truth)
    swapped = _swap(tmp_path).swapped(pool, _table(tmp_path / "v.csv", 1.0 - truth), _identifiers())

    assert swapped.names == pool.names
    assert np.array_equal(swapped.predictions[0], pool.predictions[0])
    assert np.array_equal(swapped.truth, 1.0 - truth)


def test_scored(tmp_path: Path) -> None:
    """A member that reads the truth scores 1, and the mask decides which cells it is read on."""
    truth = _alternating()
    mask = np.ones(truth.shape, dtype=np.bool_)

    assert _swap(tmp_path).scored(truth, truth, mask) == 1.0
    assert _swap(tmp_path).scored(1.0 - truth, truth, mask) == 0.0


def test_forward(tmp_path: Path) -> None:
    """The greedy seats the member that agrees with THIS truth first, whichever truth that is."""
    truth = _alternating()
    swap = _swap(tmp_path)
    pool = _pool(truth)

    assert swap.forward(pool)[0].name == "reader"
    assert swap.forward(replace(pool, truth=1.0 - truth))[0].name == "contrarian", (
        "the same predictions, the other truth — the order is a property of the ruler"
    )


def test_diff() -> None:
    """A seat that changes hands is named MOVED, and one that does not is not."""
    before = [Seat("a", 0.9), Seat("b", 0.91)]
    lines = RulerSwap.diff(before, [Seat("a", 0.8), Seat("c", 0.85)])

    assert "MOVED" in lines[2]
    assert "MOVED" not in lines[1]


def test_churn() -> None:
    """Churn reads the seated SET, so a reordering of the same members is no reseat at all."""
    before = [Seat("a", 0.9), Seat("b", 0.91)]

    assert RulerSwap.churn(before, [Seat("b", 0.9), Seat("a", 0.91)]) == 0.0
    assert RulerSwap.churn(before, [Seat("a", 0.9), Seat("c", 0.91)]) == 0.5


def test_roster_members(tmp_path: Path) -> None:
    """A roster is member names, not the prose it carries."""
    roster = tmp_path / "members.roster"
    roster.write_text("# a comment\n\nalpha\n  beta.pt  \ngamma.pt   # 0.8523 solo\n", encoding="utf-8")

    assert RulerSwap.roster_members(roster) == ["alpha", "beta", "gamma"], "a trailing comment and a .pt are both prose"
