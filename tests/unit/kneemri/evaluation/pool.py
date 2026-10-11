"""The pool that holds names, member predictions and truth together — the join every reading starts from."""

from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.scores import LabelScores
from kneemri.inference.vote import RankAverage

STUDIES = 40
SEED = 2026


def _truth() -> np.ndarray:
    """Half the studies positive on every label."""
    column = np.array([1.0] * (STUDIES // 2) + [0.0] * (STUDIES // 2), dtype=np.float32)
    return np.repeat(column[:, None], len(TARGETS), axis=1)


def _member(strength: float, seed: int) -> np.ndarray:
    """A member that ranks the positives above the negatives with the given reliability."""
    generator = np.random.default_rng(seed)
    signal = _truth()[:, 0].astype(np.float64) * strength
    return np.repeat((signal + generator.normal(0.0, 1.0, STUDIES))[:, None], len(TARGETS), axis=1)


def _names(members: list[np.ndarray]) -> list[str]:
    """Names for a pool whose test does not care which member is which."""
    return [f"m{index}.pt" for index in range(len(members))]


def test_len():
    """The member count every subset enumeration runs over."""
    members = [_member(1.0, seed) for seed in (1, 2)]
    assert len(Pool(_names(members), members, _truth())) == 2


def test_combined():
    """A subset is combined under the shipped rule, and by INDEX rather than by a copied comprehension."""
    members = [_member(1.0, seed) for seed in (1, 2, 3)]
    pool = Pool(_names(members), members, _truth())
    expected = RankAverage().combine([members[0], members[2]])
    assert np.array_equal(pool.combined((0, 2)), expected)


def test_named():
    """The subset reads as the names a submission would package, in the subset's order."""
    members = [_member(1.0, seed) for seed in (1, 2, 3)]
    assert Pool(["a.pt", "b.pt", "c.pt"], members, _truth()).named((2, 0)) == "c.pt + a.pt"


def test_rule():
    """The pool names the vote it actually combines under, so a reading cannot print one rule and apply another."""
    members = [_member(1.0, seed) for seed in (1, 2)]
    assert Pool(["a.pt", "b.pt"], members, _truth(), RankAverage()).rule() == RankAverage().named()


def test_alone():
    """Every member's own score is reported beside its name, which is what a mix is read against."""
    members = [_member(3.0, SEED), _member(0.05, SEED + 1)]
    lines = Pool(["strong.pt", "weak.pt"], members, _truth()).alone(LabelScores())

    assert [line.split(" alone")[0] for line in lines] == ["strong.pt", "weak.pt"]
    assert float(lines[0].split()[-1]) > float(lines[1].split()[-1]), "the member that follows truth scores higher"


def test_split(tmp_path: Path):
    """The roster is separated from the candidates over ONE join, so both are priced on one denominator."""
    members = [_member(1.0, seed) for seed in (1, 2, 3)]
    pool = Pool(["shipped.pt", "other.pt", "candidate.pt"], members, _truth())

    held, candidates = pool.split([tmp_path / "shipped.pt", tmp_path / "other.pt"])

    assert held.names == ["shipped.pt", "other.pt"], "both named members are held, in pool order"
    assert held.predictions[1] is members[1], "a held name keeps its own predictions"
    assert candidates.names == ["candidate.pt"], "everything not named is priced against the roster"
    assert candidates.truth.shape == pool.truth.shape, "candidates keep the join's truth, not their own"


def test_split_refuses_a_roster_member_with_no_record(tmp_path: Path):
    """A missing roster member would silently score a smaller roster and misprice every delta under it."""
    members = [_member(1.0, 1)]
    pool = Pool(["shipped.pt"], members, _truth())

    with pytest.raises(ValueError, match="no holdout record"):
        pool.split([tmp_path / "shipped.pt", tmp_path / "absent.pt"])


def test_restricted():
    """A boolean subgroup mask cuts members, truth AND the study names on the same rows."""
    members = [_member(1.0, seed) for seed in (1, 2)]
    studies = [f"s{index}" for index in range(STUDIES)]
    pool = Pool(_names(members), members, _truth(), RankAverage(), studies)
    keep = np.zeros(STUDIES, dtype=np.bool_)
    keep[:3] = True

    subgroup = pool.restricted(keep)

    assert len(subgroup) == len(pool), "a subgroup drops no member"
    assert subgroup.truth.shape[0] == 3
    assert subgroup.studies == ["s0", "s1", "s2"], "the identifiers are cut on the same rows as the arrays"
    assert np.array_equal(subgroup.predictions[1], members[1][keep])


def test_rows():
    """A subset of the studies restricts members and truth TOGETHER, which is what keeps a split honest."""
    members = [_member(1.0, seed) for seed in (1, 2)]
    pool = Pool(_names(members), members, _truth())
    studies = np.array([0, 1, 2, STUDIES - 1], dtype=np.int64)

    restricted = pool.rows(studies)

    assert len(restricted) == len(pool), "restricting studies drops no member"
    assert restricted.truth.shape[0] == len(studies)
    assert np.array_equal(restricted.predictions[1], members[1][studies]), "each member is cut on the same rows"
    assert np.array_equal(restricted.truth, pool.truth[studies])
