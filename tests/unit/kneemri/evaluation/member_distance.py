"""The distance ruler, on constructed members whose twin-vs-stranger answer is known by hand."""

import numpy as np

from core.findings import TARGETS
from kneemri.evaluation.member_curve import Pool
from kneemri.evaluation.member_distance import MemberDistance

STUDIES = 60


def _truth() -> np.ndarray:
    """Unused by this ruler by design — carried only because `Pool` holds it."""
    column = np.array([1.0] * (STUDIES // 2) + [0.0] * (STUDIES // 2), dtype=np.float32)
    return np.repeat(column[:, None], len(TARGETS), axis=1)


def _member(seed: int) -> np.ndarray:
    """An ordering drawn independently of every other seed."""
    generator = np.random.default_rng(seed)
    return generator.normal(0.0, 1.0, (STUDIES, len(TARGETS)))


def _twin_of(member: np.ndarray, seed: int) -> np.ndarray:
    """The same ordering nudged — a member that contributes an incumbent's ranking a second time."""
    generator = np.random.default_rng(seed)
    return member + generator.normal(0.0, 0.05, member.shape)


def _pool(members: list[np.ndarray]) -> Pool:
    """Members named the way a roster names them."""
    return Pool([f"m{index}.pt" for index in range(len(members))], members, _truth())


def test_agreement():
    """A member agrees perfectly with itself and near-zero with an independent draw."""
    first, second = _member(1), _member(2)
    assert MemberDistance.agreement(first, first) == 1.0
    assert abs(MemberDistance.agreement(first, second)) < 0.4


def test_agreement_is_blind_to_a_monotone_rescale():
    """Ranks, not logits — so a recalibrated member is the SAME member to this ruler."""
    member = _member(3)
    assert MemberDistance.agreement(member, member * 7.0 - 2.0) == 1.0


def test_against():
    """One member against each seat, in roster order."""
    roster = _pool([_member(1), _member(2)])
    distance = MemberDistance(roster, _pool([]))
    assert len(distance.against(_member(9))) == 2


def test_against_skips_a_seat():
    """An incumbent measured against the roster must not count itself as its own twin."""
    members = [_member(1), _member(2)]
    distance = MemberDistance(_pool(members), _pool([]))
    assert distance.against(members[0], skip=0) == [MemberDistance.agreement(members[0], members[1])]


def test_spread():
    """One scale line per seat, naming it."""
    roster = _pool([_member(1), _member(2), _member(3)])
    lines = MemberDistance(roster, _pool([])).spread()
    assert len(lines) == 3
    assert "m0.pt" in lines[0]


def test_report():
    """THE ORACLE: the board split that motivated this file, reproduced on constructed members.

    `band_span` was the better member by solo score and by seat value and cost a tick because it
    duplicated an incumbent's ordering; `lr4e-5_ws` was worse by both and took the entry. A ruler that
    cannot put the stranger first has no reason to exist.
    """
    seated = [_member(1), _member(2)]
    candidates = _pool([_twin_of(seated[0], 11), _member(12)])
    candidates.names[:] = ["twin.pt", "stranger.pt"]
    lines = MemberDistance(_pool(seated), candidates).report()
    ordered = [line for line in lines if ".pt" in line and "seated" not in line]
    assert "stranger.pt" in ordered[0]
    assert "twin.pt" in ordered[1]
