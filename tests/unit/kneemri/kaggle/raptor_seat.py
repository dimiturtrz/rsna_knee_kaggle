"""The seat blend: it must rank, it must weight, and it must join by study rather than by row."""

import numpy as np
import pandas as pd
import pytest

from core.findings import STUDY, TARGETS
from kneemri.inference.vote import Vote
from kneemri.kaggle.raptor_seat import OUR_SEATS, SEATS, RaptorSeat


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
    with pytest.raises(ValueError, match="absent from a raptor table"):
        RaptorSeat().blended(mine, theirs)


def test_a_weight_outside_a_fraction_is_refused() -> None:
    """A seat's share is in [0, 1]; anything else is a typo that would still produce a table."""
    mine = _table(["a", "b"], _rising(2))
    with pytest.raises(ValueError, match="is not one"):
        RaptorSeat(weight=1.5).blended(mine, mine)


def test_fleet() -> None:
    """Three foreign members take THREE seats of nine, not one seat split three ways.

    The distinction is the whole design: the roster they join is six members wide either way, so a
    fleet seated at 1/7 total would price three architectures as cheaply as one and read as flat for a
    reason that is arithmetic rather than diversity.
    """
    studies = ["a", "b", "c", "d"]
    mine = _table(studies, _rising(len(studies)))
    theirs: dict[str, pd.DataFrame] = {
        str(name): _table(studies, _rising(len(studies), offset=step / 10)[::-1]) for step, name in enumerate("xyz")
    }

    blended = RaptorSeat.fleet(mine, theirs)

    share = 1.0 / (OUR_SEATS + len(theirs))
    expected = (1.0 - share * len(theirs)) * Vote.ranks(mine[list(TARGETS)].to_numpy())
    for table in theirs.values():
        expected = expected + share * Vote.ranks(table[list(TARGETS)].to_numpy())
    np.testing.assert_allclose(blended[list(TARGETS)].to_numpy(), expected)


def test_combined() -> None:
    """Explicit per-table weights, and a refusal when they sum past the whole blend.

    Summing past 1 is the case worth its own assertion: it gives OUR table a negative weight, which
    reverses our own ranking and still writes a perfectly well-formed CSV.
    """
    studies = ["a", "b"]
    mine = _table(studies, _rising(len(studies)))

    blended = RaptorSeat.combined(mine, [mine], [0.25])
    np.testing.assert_allclose(blended[list(TARGETS)].to_numpy(), Vote.ranks(mine[list(TARGETS)].to_numpy()))

    with pytest.raises(ValueError, match="not a fraction"):
        RaptorSeat.combined(mine, [mine, mine], [0.7, 0.7])


def test_tilted() -> None:
    """Uniform weights equal `blended`; mixed weights equal each column's own `blended`; bad weights refused."""
    studies = ["a", "b", "c", "d"]
    mine = _table(studies, _rising(len(studies)) * np.arange(1, len(TARGETS) + 1))
    yours = _table(studies, _rising(len(studies), offset=0.05)[::-1] * np.arange(len(TARGETS), 0, -1))
    columns = list(TARGETS)

    uniform = RaptorSeat.tilted(mine, yours, dict.fromkeys(TARGETS, 0.25))
    reference = RaptorSeat(weight=0.25).blended(mine, yours)
    assert list(uniform.columns) == [STUDY, *TARGETS]
    pd.testing.assert_frame_equal(uniform, reference)

    mixed = {finding: (0.5 if index % 2 else 1.0 / 3.0) for index, finding in enumerate(TARGETS)}
    tilted = RaptorSeat.tilted(mine, yours, mixed)
    for finding, weight in mixed.items():
        expected = RaptorSeat(weight=weight).blended(mine, yours)[finding]
        np.testing.assert_allclose(tilted[finding].to_numpy(), expected.to_numpy())
    assert list(tilted[STUDY]) == studies
    assert list(tilted.columns) == [STUDY, *columns]

    with pytest.raises(ValueError, match="the board scores"):
        RaptorSeat.tilted(mine, yours, dict.fromkeys(columns[1:], 0.25))
    with pytest.raises(ValueError, match="is not one"):
        RaptorSeat.tilted(mine, yours, {**dict.fromkeys(TARGETS, 0.25), columns[0]: 1.5})


def test_a_zero_weight_returns_our_own_ranks() -> None:
    """The identity end of the axis, so the blend cannot be silently shifting our table on its own."""
    studies = ["a", "b", "c"]
    mine, yours = _rising(3), _rising(3)[::-1]
    blended = RaptorSeat(weight=0.0).blended(_table(studies, mine), _table(studies, yours))
    np.testing.assert_allclose(blended[list(TARGETS)].to_numpy(), Vote.ranks(mine))
