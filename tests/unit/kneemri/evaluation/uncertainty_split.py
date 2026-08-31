"""The reducible share, on members whose agreement is set by construction rather than measured."""

from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.member_divergence import MemberDivergence
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.uncertainty_split import CERTAIN, UncertaintyShare, UncertaintySplit
from kneemri.runs.holdout_record import HoldoutRecord

STUDIES = 40
RNG = np.random.default_rng(2026)
CLAMP_COST = 1e-4
"""What holding a certain belief off 0 and 1 costs a bit — `BOUND`'s documented, tiny price."""


def _truth() -> np.ndarray:
    """A soft reader-agreement column, the shape `RecordMix` actually builds."""
    return np.repeat(np.linspace(0.0, 1.0, STUDIES)[:, None], len(TARGETS), axis=1).astype(np.float32)


def _member(seed: int, temperature: float = 1.0) -> np.ndarray:
    """One member's beliefs, sharpened or flattened by a temperature."""
    logits = np.random.default_rng(seed).normal(0.0, 1.0, (STUDIES, len(TARGETS)))
    return 1.0 / (1.0 + np.exp(-temperature * logits))


def _pool(members: list[np.ndarray]) -> Pool:
    return Pool([f"m{index}.pt" for index in range(len(members))], members, _truth())


def _record(path: Path, member: np.ndarray) -> None:
    """A member's holdout record and the checkpoint beside it — a record alone is a dead run."""
    studies = [f"s{index}" for index in range(STUDIES)]
    HoldoutRecord(path).write(studies, member, _truth(), epoch=6)
    path.write_bytes(b"")


def test_entropy():
    """A hedged belief is a whole bit, a certain one is none, and the clamp costs no more than its price."""
    hedged = np.full((STUDIES, len(TARGETS)), 0.5)
    certain = np.ones((STUDIES, len(TARGETS)))

    assert UncertaintySplit.entropy(hedged) == pytest.approx(CERTAIN)
    assert UncertaintySplit.entropy(certain) == pytest.approx(0.0, abs=CLAMP_COST)


def test_epistemic():
    """The difference of the two entropies, which is what the whole report divides by."""
    assert UncertaintyShare(0.8, 0.5).epistemic == pytest.approx(0.3)


def test_reducible():
    """The share, and a pool with nothing to divide reads 0.0 rather than raising on 0/0."""
    assert UncertaintyShare(0.8, 0.6).reducible == pytest.approx(0.25)
    assert UncertaintyShare(0.0, 0.0).reducible == 0.0


def test_line():
    """One finding's row names itself and prints the share as a percentage, not as a bare ratio."""
    line = UncertaintyShare(0.8, 0.6).line("ACL")

    assert "ACL" in line
    assert "25.0%" in line
    assert "epistemic 0.2000 bits" in line


def test_per_label():
    """One decomposition per finding, and the aleatoric half can never exceed the total.

    Jensen's inequality is the guarantee — entropy is concave, so the entropy of the mean belief is at
    least the mean of the entropies — and a violation here would mean the two terms were computed over
    different axes, which is the arithmetic error this shape invites.
    """
    shares = UncertaintySplit(_pool([_member(1), _member(2), _member(3)])).per_label()

    assert len(shares) == len(TARGETS)
    assert all(share.epistemic >= 0.0 for share in shares)
    assert all(share.total <= CERTAIN for share in shares)


def test_per_label_matches_the_pairwise_divergence_it_generalises():
    """THE ORACLE: at K=2 the epistemic term IS `MemberDivergence.js`, the number `seat_value` prints.

    The identity `JSD(p_1..p_K) = H[mean p] - mean H[p]` is exact, so the K-way reading has to reproduce
    the pairwise one it extends — that is the whole claim this module rests on, and the two arrive
    through different arithmetic (a mixture KL there, two entropies here).
    """
    one, other = _member(1), _member(2)
    shares = UncertaintySplit(_pool([one, other])).per_label()

    pooled = float(np.mean([share.epistemic for share in shares]))
    assert pooled == pytest.approx(MemberDivergence.js(one, other), abs=CLAMP_COST)


def test_per_label_refuses_a_pool_too_small_to_disagree():
    """One member diverges from nobody, and 0.0% reducible is the one wrong answer this could give."""
    with pytest.raises(ValueError, match="no disagreement"):
        UncertaintySplit(_pool([_member(1)])).per_label()


def test_pooled():
    """A ratio of the twelve means, never a mean of twelve ratios — the flattening the shape avoids."""
    split = UncertaintySplit(_pool([_member(1), _member(2)]))
    shares = split.per_label()

    assert split.pooled().total == pytest.approx(float(np.mean([share.total for share in shares])))
    assert split.pooled().reducible != pytest.approx(float(np.mean([share.reducible for share in shares])))


def test_pooled_reads_agreement_as_nothing_to_reduce():
    """Members that believe the same thing leave no reducible share, however uncertain they are.

    The oracle for the whole module: two copies of one hedged member are maximally UNCERTAIN and
    perfectly agreed, so a ruler reading a divergence alone would say the same thing about them as
    about two confident members who disagree. The fraction is what tells those apart.
    """
    hedged = _member(1, temperature=0.01)

    assert UncertaintySplit(_pool([hedged, hedged.copy()])).pooled().total > 0.9
    assert UncertaintySplit(_pool([hedged, hedged.copy()])).pooled().reducible == pytest.approx(0.0, abs=1e-6)


def test_denominator():
    """How many members, how many studies, and that they were held out — stated, never assumed."""
    line = UncertaintySplit(_pool([_member(1), _member(2)])).denominator()

    assert "2 separately trained members" in line
    assert f"{STUDIES} studies" in line
    assert "HELD OUT" in line


def test_report():
    """The denominator, one row per finding, and the pooled share beneath them."""
    lines = UncertaintySplit(_pool([_member(1), _member(2)])).report()

    assert "HELD OUT" in lines[0]
    assert len(lines) == len(TARGETS) + 3
    assert "POOLED" in lines[-1]
    assert TARGETS[0] in lines[2]


def test_over(tmp_path: Path):
    """Members named as checkpoints become the pool their records share."""
    for index in (1, 2):
        _record(tmp_path / f"m{index}.pt", _member(index))

    split = UncertaintySplit.over([tmp_path / "m1.pt", tmp_path / "m2.pt"])

    assert len(split.pool) == 2
    assert len(split.pool.truth) == STUDIES


def test_over_refuses_a_member_that_held_nothing_out(tmp_path: Path):
    """The leak guard, and it is a refusal rather than a skip: a fold -1 member has seen every study."""
    _record(tmp_path / "m1.pt", _member(1))
    (tmp_path / "allfit_ff.pt").write_bytes(b"")

    with pytest.raises(ValueError, match="allfit_ff"):
        UncertaintySplit.over([tmp_path / "m1.pt", tmp_path / "allfit_ff.pt"])
