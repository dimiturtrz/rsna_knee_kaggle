"""Unit tests for the fp16 tie instrument.

Built on a pool whose answer is known by construction: a member that is CONSTANT ties every study, so
the two rank rules must disagree about it, and a member with distinct values ties nothing, so they must
agree exactly. Those two cases are the whole claim the instrument makes at scale.
"""

from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.pool import Pool
from kneemri.kaggle.fp16_ties import SERVING, TRAINING, Fp16Ties, TieReading

STUDIES = 120
MEMBERS = 3
SEED = 7
CHANCE = 0.5
PRECISIONS = 2


def a_pool() -> Pool:
    generator = np.random.default_rng(SEED)
    truth = (generator.random((STUDIES, len(TARGETS))) < CHANCE).astype(np.float64)
    predictions = [truth + generator.normal(0.0, 1.0, size=truth.shape) for _ in range(MEMBERS)]
    return Pool(names=[f"m{index}" for index in range(MEMBERS)], predictions=predictions, truth=truth)


def a_ties() -> Fp16Ties:
    return Fp16Ties(models=Path("never-read"), base=("a", "b", "c"))


def test_served() -> None:
    """A value fp16 cannot represent comes back changed, and one it can comes back identical."""
    served = Fp16Ties.served(np.array([[0.5, 0.1]]))
    assert served[0, 0] == 0.5, "a power of two survives the round trip"
    assert served[0, 1] != 0.1, "and a value needing more than eleven mantissa bits does not"


def test_tie_mass() -> None:
    """Ties are counted PER LABEL, so a value shared across two findings is not a tie."""
    assert Fp16Ties.tie_mass(np.array([[1.0], [2.0], [3.0]])) == 0.0
    assert Fp16Ties.tie_mass(np.array([[1.0], [1.0], [1.0]])) == 1.0
    assert Fp16Ties.tie_mass(np.array([[1.0, 1.0], [2.0, 2.0]])) == 0.0, "across labels is not a tie"


def test_ordinal() -> None:
    """The pre-fix rule: a constant column votes a full sweep ordered by corpus position."""
    assert Fp16Ties.ordinal(np.array([[3.0], [1.0], [2.0]]))[:, 0].tolist() == [1.0, 0.0, 0.5]
    assert Fp16Ties.ordinal(np.array([[1.0], [1.0], [1.0]]))[:, 0].tolist() == [0.0, 0.5, 1.0], (
        "which is the defect the fix removes"
    )


def test_reading() -> None:
    """With nothing tied the two rules are the same rule, so the cost must be exactly zero."""
    pool = a_pool()
    reading = a_ties().reading(pool.predictions, pool.truth, TRAINING)
    assert reading.precision == TRAINING
    assert reading.tie_mass == 0.0, "continuous float64 draws never tie"
    assert reading.cost() == 0.0


def test_readings() -> None:
    """Serving ties more than training does, which is the reason the instrument exists."""
    readings = a_ties().readings(a_pool())
    assert [reading.precision for reading in readings] == [TRAINING, SERVING]
    assert len(readings) == PRECISIONS
    assert readings[1].tie_mass >= readings[0].tie_mass


def test_cost() -> None:
    """The cost is what the ordinal rule LOSES, so a better midrank score is a positive number."""
    assert TieReading(precision=SERVING, tie_mass=0.1, midrank_score=0.9, ordinal_score=0.8).cost() == pytest.approx(
        0.1
    )


def test_report() -> None:
    """The report names the served cost, because that is the number a slot decision reads."""
    lines = a_ties().report(a_ties().readings(a_pool()))
    assert any("serving precision" in line for line in lines)
    assert any(line.startswith("  precision") for line in lines)
