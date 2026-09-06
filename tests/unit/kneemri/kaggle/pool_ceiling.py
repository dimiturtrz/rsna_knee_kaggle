"""Unit tests for the pool's linear ceiling.

Exercised on synthetic pools whose answer is known by construction: a pool holding a member that IS the
truth must stack far above chance, and a pool of pure noise must stack at chance — the second is the one
that catches a stack scoring itself in sample, which would report near 1.0 on noise.
"""

from pathlib import Path

import numpy as np

from kneemri.evaluation.pool import Pool
from kneemri.kaggle.pool_ceiling import PENALTIES, WEAK_COLUMNS, PoolCeiling

STUDIES = 200
LABELS = 12
MEMBERS = 4
FOLDS = 4
SEED = 7
CHANCE = 0.5
TOLERANCE = 0.12


def a_pool(*, informative: bool) -> Pool:
    generator = np.random.default_rng(SEED)
    truth = (generator.random((STUDIES, LABELS)) < CHANCE).astype(np.float64)
    predictions = [generator.random((STUDIES, LABELS)) for _ in range(MEMBERS)]
    if informative:
        predictions[0] = truth + generator.normal(0.0, 0.2, size=truth.shape)
    return Pool(names=[f"m{index}" for index in range(MEMBERS)], predictions=predictions, truth=truth)


def a_ceiling(tmp_path: Path) -> PoolCeiling:
    return PoolCeiling(tmp_path, folds=FOLDS)


def test_features(tmp_path: Path) -> None:
    pool = a_pool(informative=False)
    features = a_ceiling(tmp_path).features(pool)
    assert features.shape == (STUDIES, MEMBERS, LABELS)
    assert features.min() >= 0.0
    assert features.max() <= 1.0


def test_out_of_fold(tmp_path: Path) -> None:
    stacked = a_ceiling(tmp_path).out_of_fold(a_pool(informative=False), penalty=1.0)
    assert stacked.shape == (STUDIES, LABELS)
    assert np.isfinite(stacked).all()


def test_readings(tmp_path: Path) -> None:
    readings = a_ceiling(tmp_path).readings(a_pool(informative=False))
    assert [reading.penalty for reading in readings] == list(PENALTIES)
    assert all(reading.members == MEMBERS for reading in readings)
    assert max(reading.score for reading in readings) < CHANCE + TOLERANCE


def test_columns(tmp_path: Path) -> None:
    readings = a_ceiling(tmp_path).columns(a_pool(informative=True), penalty=1.0)
    assert len(readings) == LABELS
    assert [reading.stacked for reading in readings] == sorted(reading.stacked for reading in readings)
    assert min(reading.best_alone for reading in readings) > CHANCE + TOLERANCE
    assert all(0 < reading.positives < STUDIES for reading in readings)
    assert all(reading.interval[0] <= reading.stacked <= reading.interval[1] for reading in readings)


def test_gain(tmp_path: Path) -> None:
    readings = a_ceiling(tmp_path).columns(a_pool(informative=True), penalty=1.0)
    assert all(reading.gain == reading.stacked - reading.best_alone for reading in readings)


def test_width(tmp_path: Path) -> None:
    readings = a_ceiling(tmp_path).columns(a_pool(informative=True), penalty=1.0)
    assert all(reading.width == reading.interval[1] - reading.interval[0] for reading in readings)
    assert all(reading.width > 0.0 for reading in readings)


def test_report(tmp_path: Path) -> None:
    lines = a_ceiling(tmp_path).report(target=0.9, pool=a_pool(informative=True))
    assert lines[0].startswith(f"{MEMBERS} members, {STUDIES} shared studies")
    best = next(line for line in lines if line.startswith("best stack:"))
    assert float(best.split()[2]) > CHANCE + TOLERANCE
    assert any("first place needs" in line for line in lines)


def test_specialists(tmp_path: Path) -> None:
    ranked = a_ceiling(tmp_path).specialists(a_pool(informative=True), WEAK_COLUMNS, count=MEMBERS, floor=0.0)
    assert len(ranked) == MEMBERS
    assert {name for name, _ in ranked} == {f"m{index}" for index in range(MEMBERS)}
    assert [advantage for _, advantage in ranked] == sorted((a for _, a in ranked), reverse=True)


def test_specialists_floor(tmp_path: Path) -> None:
    pool = a_pool(informative=True)
    assert a_ceiling(tmp_path).specialists(pool, WEAK_COLUMNS, count=MEMBERS, floor=1.0) == []
