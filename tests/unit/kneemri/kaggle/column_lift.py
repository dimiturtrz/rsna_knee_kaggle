"""Unit tests for the synthetic column-lift seat.

Exercised on a pool whose answer is known by construction: the members are noise, so dragging two
columns toward truth must raise exactly those two columns and must raise the seated score monotonically.
The test that matters most is the NULL one — at strength zero the synthetic member is the blend under a
monotone remap, so seating it can only move the score by the duplicate-weight artefact (the blend then
carries two of n+1 votes), which must be negligible. A lift that silently rescales every column, or a
rank map that is not monotone, shows up there as a gain the columns did not earn.
"""

from pathlib import Path

import numpy as np

from core.findings import TARGETS
from kneemri.evaluation.pool import Pool
from kneemri.kaggle.column_lift import STRENGTHS, ColumnLift, LiftReading
from kneemri.kaggle.pool_ceiling import WEAK_COLUMNS

STUDIES = 300
MEMBERS = 3
SEED = 11
CHANCE = 0.5
NULL_SEAT = 1e-3
ORACLE = 1.0
TARGET = 0.01


def a_pool() -> Pool:
    generator = np.random.default_rng(SEED)
    truth = (generator.random((STUDIES, len(TARGETS))) < CHANCE).astype(np.float64)
    predictions = [truth + generator.normal(0.0, 1.0, size=truth.shape) for _ in range(MEMBERS)]
    return Pool(names=[f"m{index}" for index in range(MEMBERS)], predictions=predictions, truth=truth)


def a_lift() -> ColumnLift:
    return ColumnLift(models=Path("never-read"), base=("a", "b", "c"))


def test_ranked() -> None:
    ranked = ColumnLift.ranked(np.array([3.0, 1.0, 2.0]))
    assert list(ranked) == [1.0, 0.0, 0.5]


def test_lifted() -> None:
    pool = a_pool()
    lift = a_lift()
    untouched = TARGETS.index(next(target for target in TARGETS if target not in WEAK_COLUMNS))
    blend = pool.combined(list(range(len(pool))))
    lifted = lift.lifted(pool, ORACLE)
    assert np.allclose(lifted[:, untouched], blend[:, untouched])
    for finding in WEAK_COLUMNS:
        assert np.allclose(lifted[:, TARGETS.index(finding)], (pool.truth[:, TARGETS.index(finding)] > CHANCE))


def test_readings() -> None:
    readings = a_lift().readings(a_pool())
    assert [reading.strength for reading in readings] == list(STRENGTHS)
    assert abs(readings[0].gain) < NULL_SEAT
    assert all(column == 1.0 for column in readings[-1].columns)
    assert readings[-1].gain > readings[0].gain


def test_weakest() -> None:
    assert LiftReading(strength=0.5, columns=(0.9, 0.7), gain=0.0).weakest() == 0.7


def test_required() -> None:
    readings = [
        LiftReading(strength=0.0, columns=(0.5, 0.5), gain=0.0),
        LiftReading(strength=1.0, columns=(1.0, 1.0), gain=0.05),
    ]
    assert ColumnLift.required(readings, TARGET) is readings[1]
    assert ColumnLift.required(readings, ORACLE) is None


def test_report() -> None:
    readings = a_lift().readings(a_pool())
    lines = a_lift().report(target=TARGET, readings=readings)
    assert any("ORACLE CEILING" in line for line in lines)
    assert any(line.startswith("  strength") for line in lines)
    refused = a_lift().report(target=ORACLE, readings=readings)
    assert any(line.startswith("REFUSED") for line in refused)
