"""MemberAxes places a candidate in the cell all three rulers agree on, on the roster's own scale."""

import numpy as np
import pytest

from kneemri.evaluation.member_axes import MemberAxes, MemberReading
from kneemri.evaluation.pool import Pool

STUDIES = 60
TARGETS = 3
SCALE = (0.5, 0.2, 1.0)


def _member(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((STUDIES, TARGETS))


def _twin_of(base: np.ndarray, seed: int) -> np.ndarray:
    return np.clip(base + np.random.default_rng(seed).normal(0.0, 0.002, base.shape), 0.0, 1.0)


@pytest.fixture(name="axes")
def _axes() -> MemberAxes:
    seated = [_member(1), _member(2), _member(3)]
    roster = Pool(["a", "b", "c"], seated, np.zeros((STUDIES, TARGETS)))
    candidates = Pool(["twin_of_a", "loner"], [_twin_of(seated[0], 9), _member(7)], np.zeros((STUDIES, TARGETS)))
    return MemberAxes(roster, candidates)


def test_celled() -> None:
    assert MemberAxes.celled(0.9, 0.1, 0.5, SCALE) == "TWIN"
    assert MemberAxes.celled(0.9, 0.9, 0.5, SCALE) == "SCALED"
    assert MemberAxes.celled(0.1, 0.9, 0.5, SCALE) == "NOISY"
    assert MemberAxes.celled(0.1, 0.9, 9.0, SCALE) == "DISTINCT"
    assert MemberAxes.celled(0.9, 0.1, 9.0, SCALE) == "PUZZLE"


def test_whitened(axes: MemberAxes) -> None:
    seated, candidates = axes.whitened()
    assert len(seated) == 3
    assert len(candidates) == 2
    assert all(len(row) == 3 for row in candidates)


def test_scale(axes: MemberAxes) -> None:
    rank, bits, span = axes.scale()
    assert -1.0 <= rank <= 1.0
    assert bits >= 0.0
    assert span > 0.0


def test_readings(axes: MemberAxes) -> None:
    readings = axes.readings()
    assert [reading.name for reading in readings] == sorted(
        [reading.name for reading in readings], key=lambda name: dict((r.name, r.agreement) for r in readings)[name]
    )
    twin = next(reading for reading in readings if reading.name == "twin_of_a")
    assert twin.cell in {"TWIN", "SCALED"}


def test_report(axes: MemberAxes) -> None:
    lines = axes.report()
    assert "member axes" in lines[0]
    assert any("ADOPTION GATE" in line for line in lines)


def test_line() -> None:
    assert "DISTINCT" in MemberReading("x", 0.5, 0.1, 2.0, "DISTINCT").line()
