"""The confidence ruler, on constructed members whose rank correlation is 1.0 by construction."""

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.member_distance import MemberDistance
from kneemri.evaluation.member_divergence import MemberDivergence
from kneemri.evaluation.pool import Pool

STUDIES = 60
PRESENT_STUDIES = STUDIES // 3
"""How many fixture studies sit at or above `PRESENT`; another third sit below it but above zero."""
SATURATED = 4.0
HEDGED = 0.15
IDENTICAL_RANKING = 0.999
CLAMP_COST = 1e-4
"""What holding a certain belief off 0 and 1 costs the bound — `BOUND`'s documented, tiny price."""


def _truth() -> np.ndarray:
    """Read only by `labelled`, and only for its support counts.

    SOFT, like the real one: `RecordMix` builds a reader-agreement scale, not a binary label, so a
    fixture of ones and zeros would let a `> 0` count pass a test that the real holdout fails.
    """
    column = np.array([1.0] * PRESENT_STUDIES + [0.28] * PRESENT_STUDIES + [0.0] * PRESENT_STUDIES)
    return np.repeat(column[:, None], len(TARGETS), axis=1)


def _logits(seed: int) -> np.ndarray:
    """An ordering drawn independently of every other seed."""
    return np.random.default_rng(seed).normal(0.0, 1.0, (STUDIES, len(TARGETS)))


def _member(seed: int, temperature: float = 1.0) -> np.ndarray:
    """That ordering as sigmoid probabilities, sharpened or flattened by a temperature."""
    return 1.0 / (1.0 + np.exp(-temperature * _logits(seed)))


def _pool(members: list[np.ndarray]) -> Pool:
    """Members named the way a roster names them."""
    return Pool([f"m{index}.pt" for index in range(len(members))], members, _truth())


def test_js():
    """Identical beliefs diverge zero, opposite certainties diverge by the bound, and it is symmetric."""
    member = _member(1)
    certain_yes = np.full((STUDIES, len(TARGETS)), 1.0)
    certain_no = np.zeros((STUDIES, len(TARGETS)))

    assert MemberDivergence.js(member, member) == 0.0
    assert MemberDivergence.js(certain_yes, certain_no) == pytest.approx(1.0, abs=CLAMP_COST), (
        "one bit is the bound, and the clamp holds the extreme case that far below it and no further"
    )
    assert MemberDivergence.js(member, _member(2)) == MemberDivergence.js(_member(2), member)


def test_kl():
    """The clamped logarithm the JS is built from, and the identity `UncertaintySplit` reads entropy by.

    Public since that module, which takes a Bernoulli entropy as `1 bit - KL(p || 0.5)` rather than
    spelling the logarithms a second time — so the identity is asserted here, where the arithmetic is.
    """
    member = _member(1)
    uniform = np.full_like(member, 0.5)

    assert np.allclose(MemberDivergence.kl(member, member), 0.0)
    entropy = -member * np.log2(member) - (1.0 - member) * np.log2(1.0 - member)
    assert np.allclose(1.0 - MemberDivergence.kl(member, uniform), entropy)


def test_js_sees_what_rank_correlation_cannot():
    """THE ORACLE: same ordering, different confidence — the case that motivates the whole module.

    A temperature applied to one member's logits leaves every study's RANK untouched, so the rank ruler
    reports a twin. The beliefs are far apart, which is what a logit average combines and what this
    ruler is here to see.
    """
    hedged, saturated = _member(1, HEDGED), _member(1, SATURATED)

    assert MemberDistance(_pool([hedged]), _pool([])).against(saturated)[0] > IDENTICAL_RANKING
    assert MemberDivergence.js(hedged, saturated) > MemberDivergence.js(hedged, _member(2))


def test_against():
    """One member against each seat, in roster order."""
    roster = _pool([_member(1), _member(2)])

    assert len(MemberDivergence(roster, _pool([])).against(_member(9))) == 2


def test_against_skips_a_seat():
    """An incumbent measured against the roster must not count its own 0.0 as its nearest neighbour."""
    members = [_member(1), _member(2)]
    divergence = MemberDivergence(_pool(members), _pool([]))

    assert divergence.against(members[0], skip=0) == [MemberDivergence.js(members[0], members[1])]


def test_spread():
    """One scale line per seat, naming it, and reading the MINIMUM rather than the maximum."""
    roster = _pool([_member(1), _member(2), _member(3)])
    lines = MemberDivergence(roster, _pool([])).spread()

    assert len(lines) == 3
    assert "m0.pt" in lines[0]
    assert f"{min(MemberDivergence(roster, _pool([])).against(roster.predictions[0], skip=0)):.4f}" in lines[0]


def test_spread_of_a_roster_with_no_neighbour():
    """A member alone diverges from nobody, and the scale says so rather than inventing one."""
    lines = MemberDivergence(_pool([_member(1)]), _pool([])).spread()

    assert len(lines) == 1
    assert "no spread" in lines[0]
    assert "roster of 1" in lines[0], "with the denominator that makes it a fact"


def test_labelled():
    """One line per label, each carrying the positive count knee-c2ew requires beside a divergence."""
    roster = _pool([_member(1), _member(2), _member(3)])
    lines = MemberDivergence(roster, _pool([])).labelled()

    assert len(lines) == len(TARGETS) + 1
    assert f"support {PRESENT_STUDIES}/{STUDIES} at >= 0.5" in lines[1], (
        "the support is a count over a stated denominator at a stated threshold, and the third of the "
        "fixture that is non-zero but under it does NOT count as present — the defect the real holdout "
        "showed on this module's first run, where a bare > 0 read 99% support on every label"
    )


def test_report():
    """Candidates ordered most divergent first, with the roster's own spread above them as the scale."""
    seated = [_member(1), _member(2)]
    candidates = _pool([_member(1, SATURATED), _member(1)])
    candidates.names[:] = ["rescaled.pt", "copy.pt"]

    lines = MemberDivergence(_pool(seated), candidates).report()

    ordered = [line for line in lines if ".pt" in line and "seated" not in line]
    assert "rescaled.pt" in ordered[0]
    assert "copy.pt" in ordered[1], "a byte-identical seat is the nearest thing a candidate can be"
    assert any("Jensen-Shannon" in line for line in lines), "the report names the metric it prints"
