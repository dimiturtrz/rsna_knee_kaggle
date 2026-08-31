"""MemberSpace ranks each member per label and centres the pool — the coordinates every ruler reads."""

import numpy as np
import pytest

from kneemri.evaluation.member_space import MemberSpace
from kneemri.evaluation.pool import Pool

STUDIES = 40
TARGETS = 3
MEMBERS = 3


@pytest.fixture(name="space")
def _space() -> MemberSpace:
    predictions = [np.random.default_rng(seed).random((STUDIES, TARGETS)) for seed in range(MEMBERS)]
    return MemberSpace(Pool(["one", "two", "three"], predictions, np.zeros((STUDIES, TARGETS))))


def test_ranked() -> None:
    ranks = MemberSpace.ranked(np.array([[0.9], [0.1], [0.5]]))
    assert ranks.min() == 0.0
    assert ranks.max() == 1.0
    assert ranks[1, 0] < ranks[2, 0] < ranks[0, 0]


def test_matrix(space: MemberSpace) -> None:
    matrix = space.matrix()
    assert matrix.shape == (MEMBERS, STUDIES * TARGETS)
    assert np.allclose(matrix.mean(axis=0), 0.0)
