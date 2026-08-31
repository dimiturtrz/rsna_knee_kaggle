"""MemberMahalanobis whitens the pool's own coordinates, and says which verdicts the shrinkage knob moves."""

import numpy as np
import pytest

from kneemri.evaluation.member_mahalanobis import MemberMahalanobis
from kneemri.evaluation.member_space import MemberSpace
from kneemri.evaluation.pool import Pool

STUDIES = 40
TARGETS = 3


def _member(seed: int, twin: int | None = None) -> np.ndarray:
    generator = np.random.default_rng(seed if twin is None else twin)
    base = generator.random((STUDIES, TARGETS))
    return base if twin is None else base + np.random.default_rng(seed).random((STUDIES, TARGETS)) * 0.001


@pytest.fixture(name="space")
def _space() -> MemberSpace:
    predictions = [_member(1), _member(2), _member(3), _member(4, twin=1)]
    names = ["one", "two", "three", "one_twin"]
    return MemberSpace(Pool(names, predictions, np.zeros((STUDIES, TARGETS))))


@pytest.fixture(name="ruler")
def _ruler(space: MemberSpace) -> MemberMahalanobis:
    return MemberMahalanobis(space)


def test_coordinates(ruler: MemberMahalanobis) -> None:
    points = ruler.coordinates()
    assert points.shape[0] == 4
    assert points.shape[1] <= 3


def test_distances(ruler: MemberMahalanobis) -> None:
    matrix = ruler.distances()
    assert np.isinf(np.diag(matrix)).all()
    assert np.allclose(matrix, matrix.T)


def test_nearest(ruler: MemberMahalanobis) -> None:
    pairs = ruler.nearest()
    assert len(pairs) == 4
    assert pairs == sorted(pairs)
    assert {pairs[0][1], pairs[0][2]} == {"one", "one_twin"}


def test_sensitivity(ruler: MemberMahalanobis) -> None:
    orderings = ruler.sensitivity((0.1, 0.9))
    assert set(orderings) == {0.1, 0.9}
    assert all(sorted(order) == sorted(ruler.space.pool.names) for order in orderings.values())


def test_unstable(ruler: MemberMahalanobis) -> None:
    assert ruler.unstable((0.2, 0.2)) == []


def test_report(ruler: MemberMahalanobis) -> None:
    lines = ruler.report()
    assert "member mahalanobis" in lines[0]
    assert any("ADOPTION GATE" in line for line in lines)
