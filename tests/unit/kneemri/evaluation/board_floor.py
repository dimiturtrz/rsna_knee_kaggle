"""What a board difference is worth at four hundred studies: the mirror of `kneemri/evaluation/board_floor.py`.

The pool here is built by hand rather than loaded, because the subject is the WIDTH of a difference and
a real pool would make the test a statement about particular members.
"""

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.board_floor import BoardFloor
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.scores import LabelScores

STUDIES = 200
LABELS = len(TARGETS)


def _pool() -> Pool:
    generator = np.random.default_rng(7)
    truth = generator.integers(0, 2, size=(STUDIES, LABELS)).astype(np.float64)
    signal = truth + generator.normal(0.0, 0.6, size=truth.shape)
    noise = generator.normal(0.0, 1.0, size=truth.shape)
    return Pool(["good", "also_good", "noise"], [signal, signal + 0.01 * noise, noise], truth)


def _floor(first: list[str], second: list[str]) -> BoardFloor:
    pool = _pool()
    return BoardFloor(pool, (BoardFloor.seats(pool, first), BoardFloor.seats(pool, second)), LabelScores())


def test_seats():
    """Names resolve to pool positions, and a name the pool never carried is refused rather than dropped."""
    pool = _pool()

    assert BoardFloor.seats(pool, ["noise", "good"]) == [2, 0]
    with pytest.raises(ValueError, match="does not carry"):
        BoardFloor.seats(pool, ["absent"])


def test_delta():
    """The difference is read on the rows it is given, and an arm against itself is exactly zero."""
    floor = _floor(["good", "noise"], ["good"])
    rows = np.arange(STUDIES)

    assert _floor(["good"], ["good"]).delta(rows) == 0.0, "the same members voted twice cannot differ"
    assert floor.delta(rows) < 0.0, "a pure-noise seat drags a good member down on this pool"


def test_interval():
    """A smaller scored sample gives a WIDER interval — which is the whole reason this module exists."""
    floor = _floor(["good", "noise"], ["good"])

    narrow = floor.interval(STUDIES, draws=120)
    wide = floor.interval(STUDIES // 4, draws=120)

    assert wide[1] - wide[0] > narrow[1] - narrow[0], "a quarter of the studies reads a wider delta"
    assert narrow[0] < floor.delta(np.arange(STUDIES)) < narrow[1], "and the observed delta sits inside its own"


def test_lines():
    """The reading names the observed delta, the board's rounding floor, and a half-width per size."""
    lines = _floor(["good", "noise"], ["good"]).lines([50], draws=80)

    assert lines[0].startswith("observed delta on our 200 studies:")
    assert "0.0005 is unreadable by rounding" in lines[1]
    assert "n=50:" in lines[2] and "half-width" in lines[2]
