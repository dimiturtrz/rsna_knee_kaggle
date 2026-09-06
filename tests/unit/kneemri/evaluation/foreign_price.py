import numpy as np
import pandas as pd

from core.findings import STUDY, TARGETS
from kneemri.evaluation.foreign_price import ForeignPrice
from kneemri.evaluation.foreign_split import ForeignSplit

TRUTH = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]], np.float32)


def _price(predictions: np.ndarray) -> ForeignPrice:
    split = ForeignSplit(fold=np.array([0, 0, 1, 1]), studies=list("abcd"))
    return ForeignPrice(predictions=predictions.astype(np.float32), truth=TRUTH, split=split)


def test_macro() -> None:
    """A perfectly ordered label scores 1.0; a constant label is nan and is dropped, never averaged as 0.5."""
    scores = np.array([[0.9, 0.1], [0.1, 0.2], [0.8, 0.3], [0.2, 0.4]], np.float32)
    assert ForeignPrice.macro(scores, TRUTH) == 1.0
    assert np.isnan(ForeignPrice.macro(scores, np.zeros((4, 2), np.float32)))


def test_solo() -> None:
    """Fold 0 is read on its own two studies only — the two it trained on cannot move its number."""
    good = np.array([[0.9, 0.1], [0.1, 0.2], [0.0, 0.0], [0.0, 0.0]], np.float32)
    bad = np.array([[0.1, 0.9], [0.9, 0.8], [0.0, 0.0], [0.0, 0.0]], np.float32)
    price = _price(np.stack([good, bad]))
    assert price.solo(0) == 1.0, "fold 0 orders its own held-out pair perfectly"
    assert price.solo(1) != price.solo(0), "fold 1 is read on the other pair, where it is not this array"


def test_matched() -> None:
    """Each row comes from the fold that held that study out; the contaminated rows never appear."""
    fold0 = np.array([[0.9, 0.1], [0.1, 0.2], [9.0, 9.0], [9.0, 9.0]], np.float32)
    fold1 = np.array([[9.0, 9.0], [9.0, 9.0], [0.8, 0.4], [0.2, 0.9]], np.float32)
    matched = _price(np.stack([fold0, fold1])).matched()
    assert matched.shape == (4, 2)
    assert 9.0 not in matched, "a cell from a fold that trained on that study is not a reading of it"
    assert np.allclose(matched[2], [0.8, 0.4])


def test_pooled() -> None:
    """Each study takes the row of the ONE fold that held it out, and every study is covered."""
    fold0 = np.array([[0.9, 0.1], [0.1, 0.2], [9.0, 9.0], [9.0, 9.0]], np.float32)
    fold1 = np.array([[9.0, 9.0], [9.0, 9.0], [0.8, 0.4], [0.2, 0.9]], np.float32)
    assert _price(np.stack([fold0, fold1])).pooled() == 1.0, "the 9.0 cells are the contaminated ones, unused"


def test_lines() -> None:
    """The reading names both denominators and refuses to print their cross-fold rank-mean."""
    lines = _price(np.zeros((5, 4, 2))).lines()
    assert "out-of-sample studies" in lines[0]
    assert "over all 4 studies, none contaminated" in lines[5]
    assert "rank-mean" in lines[6]


def test_truth_of() -> None:
    """No holdout npz means the training table IS the truth, taken in the served order and deduplicated."""
    table = pd.DataFrame(
        [
            {STUDY: study, "Report": "r", **{target: float(index) for target in TARGETS}}
            for index, study in enumerate("ab")
        ]
    )
    doubled = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    truth = ForeignPrice.truth_of(None, ["b", "a"], doubled)
    assert truth.shape == (2, len(TARGETS))
    assert truth[0, 0] == 1.0, "the served order decides the rows, not the table's"
    assert truth[1, 0] == 0.0
