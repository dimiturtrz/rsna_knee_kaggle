"""The census, on a pool built so that exactly one member is strong AND distant by construction."""

import numpy as np

from core.findings import TARGETS
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.pool_census import STRONG_OF_BEST, MemberCensus, PoolCensus
from kneemri.evaluation.scores import LabelScores

STUDIES = 200
TWIN_NOISE = 1e-3
"""How far the twin sits from the member it copies — far enough to be a second array, near enough to rank with it."""
WEAK_SIGNAL = 0.15
"""How much of the truth the weak member is given, so it is distant for the uninteresting reason."""


def _truth() -> np.ndarray:
    """Half the studies positive on every label, as the float the holdout record stores."""
    column = np.array([1.0] * (STUDIES // 2) + [0.0] * (STUDIES // 2), dtype=np.float32)
    return np.repeat(column[:, None], len(TARGETS), axis=1)


def _member(seed: int, signal: float = 1.0) -> np.ndarray:
    """Truth plus noise, squashed — a member whose score rises with how much truth it is given."""
    rng = np.random.default_rng(seed)
    return 1.0 / (1.0 + np.exp(-(signal * (_truth() * 2.0 - 1.0) + rng.normal(0.0, 1.0, _truth().shape))))


def _twin(member: np.ndarray, seed: int) -> np.ndarray:
    """A near-copy that ranks ALMOST identically — jittered, not shifted.

    A constant offset preserves every rank exactly, so a twin built that way reads agreement 1.0 and the
    census cannot be caught reporting a member as its own neighbour. The jitter is what makes the twin a
    second member rather than the same one twice.
    """
    return member + np.random.default_rng(seed).normal(0.0, TWIN_NOISE, member.shape)


def _reordered(member: np.ndarray, seed: int) -> np.ndarray:
    """The same member with its values shuffled WITHIN each class, per label.

    THE FIXTURE'S WHOLE POINT. Area under the curve counts positive-negative pairs in the right order, so
    permuting values inside the positive group and inside the negative group leaves the score EXACTLY
    unchanged while destroying the rank correlation with the member it came from. That is a
    strong-and-distant member by construction rather than by a lucky seed.
    """
    rng = np.random.default_rng(seed)
    truth, shuffled = _truth(), member.copy()
    for label in range(member.shape[1]):
        for value in (1.0, 0.0):
            rows = np.flatnonzero(truth[:, label] == value)
            shuffled[rows, label] = member[rng.permutation(rows), label]
    return shuffled


def _pool(members: list[np.ndarray], names: list[str]) -> Pool:
    """Members under the names a census prints."""
    return Pool(names, members, _truth())


def _twinned() -> Pool:
    """A member, its near-twin, and a third that scores the same while ranking differently."""
    member = _member(1)
    return _pool(
        [member, _twin(member, 3), _reordered(member, 7)],
        ["member.pt", "twin.pt", "reordered.pt"],
    )


def _census(pool: Pool) -> PoolCensus:
    """The census under the project's own scorer."""
    return PoolCensus(pool, LabelScores())


def test_line():
    """The row names the member and prints all three coordinates."""
    line = MemberCensus("a.pt", 0.8642, 0.8772, 0.0665).line()

    assert "a.pt" in line
    assert "0.8642" in line
    assert "0.8772" in line
    assert "0.0665" in line


def test_solo():
    """A member given more of the truth scores higher, which is the axis `strong` is read on."""
    pool = _twinned()

    assert _census(pool).solo(_member(1)) > _census(pool).solo(_member(1, WEAK_SIGNAL))


def test_rows():
    """One row per member, each measured against the OTHERS rather than against itself."""
    pool = _twinned()
    rows = _census(pool).rows()

    assert [row.name for row in rows] == pool.names
    assert rows[0].agreement < 1.0, "a member skipping its own seat cannot be its own nearest twin"
    assert rows[0].agreement > rows[2].agreement, "the twin pair are nearer each other than the reordered member"


def test_occupants():
    """THE ORACLE: the reordered member scores what its original scores and ranks nothing like it.

    It is strong by construction — a within-class permutation leaves the area under the curve exactly
    unchanged — and distant by construction, so it is the cell's occupant while the twin pair are not.
    """
    rows = _census(_twinned()).rows()
    occupants = _census(_twinned()).occupants(rows)

    assert [row.name for row in occupants] == ["reordered.pt"]
    assert abs(rows[0].solo - rows[2].solo) < STRONG_OF_BEST, "the permutation preserved the score"


def test_occupants_when_the_distant_member_is_merely_weak():
    """The cell is EMPTY when the only distant member is distant because it is bad — the real pool's case.

    knee-nj4l's note is exactly this shape: `mae_matched` and `focus_mcl` are decorrelated because they
    are undertrained and narrow. A census that counted them would answer the bead's question with the
    two members that cannot answer it.
    """
    member = _member(1)
    pool = _pool([member, _twin(member, 3), _member(9, WEAK_SIGNAL)], ["a.pt", "b.pt", "weak.pt"])
    census = _census(pool)

    assert census.occupants(census.rows()) == []


def test_verdict():
    """The verdict prints both cuts and the denominator they were read off."""
    census = _census(_twinned())
    lines = census.verdict(census.rows())

    assert "over 3 members" in lines[0]
    assert "this pool's median" in lines[0]
    assert "reordered.pt" in lines[1]


def test_verdict_of_an_empty_cell():
    """An empty cell says what it is evidence OF, since that is the outcome worth stating carefully."""
    member = _member(1)
    census = _census(_pool([member, _twin(member, 3), _member(9, WEAK_SIGNAL)], ["a.pt", "b.pt", "weak.pt"]))
    lines = census.verdict(census.rows())

    assert "EMPTY" in lines[1]
    assert "unrepresented" in lines[1], "an empty cell is not a refutation, and the wording carries that"


def test_report():
    """The cell first, the ranked table after it, and the table says how much of the pool it shows."""
    lines = _census(_twinned()).report()

    assert "strong-and-distant cell" in lines[0]
    assert any("best first (showing 3 of 3)" in line for line in lines)


def test_report_of_a_pool_too_small_to_have_a_median():
    """Two members have one pairing, so a below-the-median cut would split one number against itself."""
    member = _member(1)
    lines = _census(_pool([member, _twin(member, 3)], ["a.pt", "b.pt"])).report()

    assert len(lines) == 1
    assert "NO CENSUS" in lines[0]
    assert "2 members" in lines[0], "with the denominator that makes it a fact"


def test_spread():
    """Both ranges, each with the count it was read over — the reading that does not rest on the cut."""
    census = _census(_twinned())
    rows = census.rows()
    lines = census.spread(rows, max(row.solo for row in rows))

    assert "among the 3 strong" in lines[0]
    assert "across all 3" in lines[1]
