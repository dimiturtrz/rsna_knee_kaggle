"""The seat blend: it must rank, it must weight, and it must join by study rather than by row."""

import numpy as np
import pandas as pd
import pytest

from core.findings import STUDY, TARGETS
from kneemri.inference.vote import Vote
from kneemri.kaggle.raptor_seat import SEATS, RaptorSeat


def _table(studies: list[str], values: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=list(TARGETS))
    frame.insert(0, STUDY, studies)
    return frame


def _rising(studies: int, offset: float = 0.0) -> np.ndarray:
    column = np.linspace(0.1, 0.9, studies) + offset
    return np.repeat(column[:, None], len(TARGETS), axis=1)


def test_blended() -> None:
    """The weighted rank average, checked against the same ranking the shipped vote uses."""
    studies = ["a", "b", "c", "d"]
    mine, yours = _rising(len(studies)), _rising(len(studies))[::-1]
    blended = RaptorSeat().blended(_table(studies, mine), _table(studies, yours))

    weight = 1.0 / SEATS
    expected = (1.0 - weight) * Vote.ranks(mine) + weight * Vote.ranks(yours)
    assert list(blended[STUDY]) == studies
    assert list(blended.columns) == [STUDY, *TARGETS]
    np.testing.assert_allclose(blended[list(TARGETS)].to_numpy(), expected)


def test_joins_by_study_not_by_position() -> None:
    """A shuffled raptor table must blend the same as an aligned one — the failure that lands near chance."""
    studies = ["a", "b", "c", "d"]
    mine, yours = _rising(len(studies)), _rising(len(studies), offset=0.05)[::-1]
    aligned = RaptorSeat().blended(_table(studies, mine), _table(studies, yours))
    shuffled = RaptorSeat().blended(_table(studies, mine), _table(studies, yours).iloc[[2, 0, 3, 1]])
    np.testing.assert_allclose(aligned[list(TARGETS)].to_numpy(), shuffled[list(TARGETS)].to_numpy())


def test_a_missing_study_is_refused() -> None:
    """Half a blend is not a submission; the roster's own table would silently carry the gap."""
    mine = _table(["a", "b", "c"], _rising(3))
    theirs = _table(["a", "b"], _rising(2))
    with pytest.raises(ValueError, match="absent from the raptor table"):
        RaptorSeat().blended(mine, theirs)


def test_a_weight_outside_a_fraction_is_refused() -> None:
    """A seat's share is in [0, 1]; anything else is a typo that would still produce a table."""
    mine = _table(["a", "b"], _rising(2))
    with pytest.raises(ValueError, match="is not one"):
        RaptorSeat(weight=1.5).blended(mine, mine)


def test_a_zero_weight_returns_our_own_ranks() -> None:
    """The identity end of the axis, so the blend cannot be silently shifting our table on its own."""
    studies = ["a", "b", "c"]
    mine, yours = _rising(3), _rising(3)[::-1]
    blended = RaptorSeat(weight=0.0).blended(_table(studies, mine), _table(studies, yours))
    np.testing.assert_allclose(blended[list(TARGETS)].to_numpy(), Vote.ranks(mine))
