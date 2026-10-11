"""A seat's price at each weight, on a pool small enough to check the arithmetic by hand."""

import numpy as np

from core.findings import TARGETS
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.scores import LabelScores
from kneemri.evaluation.seat_weight import SeatWeight
from kneemri.inference.vote import RankAverage


def pool_of() -> Pool:
    """Two base members that rank the studies backwards and a candidate that ranks them right."""
    columns = len(TARGETS)
    truth = np.tile(np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32), (1, columns))
    wrong = np.tile(np.array([[0.9], [0.8], [0.2], [0.1]], dtype=np.float32), (1, columns))
    right = np.tile(np.array([[0.1], [0.2], [0.8], [0.9]], dtype=np.float32), (1, columns))
    return Pool(
        names=["wrong_a", "wrong_b", "candidate"],
        predictions=[wrong, wrong.copy(), right],
        truth=truth,
        vote=RankAverage(),
    )


def seat_of() -> SeatWeight:
    """The candidate seated last, priced against the two members that disagree with it."""
    return SeatWeight(pool_of(), seat=2, scores=LabelScores())


def test_scores_defaults_to_ledger75() -> None:
    """The unspecified cut is LEDGER75, so a weight is comparable to a `seat_add` number."""
    assert SeatWeight(pool_of(), seat=2).scores.name() == "ledger75"
    assert LabelScores().name() != "ledger75", "the midpoint has to stay reachable and distinguishable"


def test_base() -> None:
    """The base is every member but the candidate, in the pool's own order."""
    assert seat_of().base() == [0, 1]


def test_blended() -> None:
    """Weight is repetition: at 2x the candidate outvotes two members that agree with each other."""
    seat = seat_of()
    assert seat.blended(0).shape == (4, len(TARGETS))
    assert not np.allclose(seat.blended(1), seat.blended(2))


def test_scored() -> None:
    """A candidate that is right against a base that is wrong lifts the score as its weight rises."""
    seat = seat_of()
    assert seat.scored(0) < seat.scored(2)


def test_interval() -> None:
    """The paired interval brackets the delta it is taken around."""
    seat = seat_of()
    low, high = seat.interval(2, draws=32, seed=1)
    assert low <= seat.scored(2) - seat.scored(0) <= high


def test_interval_is_pinned() -> None:
    """Bit-equal to the hand-written nan-tolerant tail, on a noisy pool where the delta has a real spread."""
    generator = np.random.default_rng(5)
    truth = (generator.random((40, len(TARGETS))) > 0.5).astype(np.float32)
    members = [(truth + generator.normal(0.0, scale, truth.shape)).astype(np.float32) for scale in (1.0, 1.2, 0.8)]
    pool = Pool(names=["a", "b", "c"], predictions=members, truth=truth, vote=RankAverage())

    interval = SeatWeight(pool, seat=2, scores=LabelScores()).interval(2, draws=200, seed=3)

    assert interval == (0.023942249062458846, 0.08379659116618934)


def test_on() -> None:
    """Reading the seat on a subset keeps the members and narrows the studies."""
    subset = seat_of().on(np.array([0, 2], dtype=np.int64))
    assert subset.pool.names == ["wrong_a", "wrong_b", "candidate"]
    assert len(subset.pool.truth) == 2


def test_split_half() -> None:
    """Two rows, one per half, each naming the weight the other half chose."""
    rows = seat_of().split_half([1, 2])
    assert len(rows) == 3
    assert "out-of-sample" in rows[1]


def test_lines() -> None:
    """One line for the base and one per weight asked for."""
    lines = seat_of().lines([1, 2])
    assert len(lines) == 3
    assert "candidate at 2x" in lines[2]


def test_rules() -> None:
    """Both shipped vote rules are offered, each under the name its own class answers to."""
    assert set(SeatWeight.rules()) == {"LogitAverage", "RankAverage"}
