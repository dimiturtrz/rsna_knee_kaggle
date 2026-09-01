"""The mirror for the epoch's draw policy: every public method called, and asserted."""

from pathlib import Path

import numpy as np
import pytest

from kneemri.training.sampling import Disagreed, Uniform

STUDIES = 40


def _positions() -> np.ndarray:
    return np.arange(STUDIES, dtype=np.int64)


def _rising() -> Disagreed:
    """Disagreement rising with the study index, at half tilt."""
    return Disagreed(np.linspace(0.01, 0.30, STUDIES), alpha=0.5)


def test_uniform_drawn() -> None:
    positions = _positions()
    generator = np.random.default_rng(0)
    assert np.array_equal(Uniform().drawn(positions, generator), positions)
    assert generator.integers(0, 100) == np.random.default_rng(0).integers(0, 100), "the stream must not advance"


def test_uniform_tilt() -> None:
    assert Uniform().tilt() == 0.0


def test_uniform_described() -> None:
    assert "uniform" in Uniform().described()


def test_weights() -> None:
    weights = _rising().weights(_positions())
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] < weights[-1]


def test_disagreed_weights_at_zero_alpha_are_flat() -> None:
    flat = Disagreed(np.linspace(0.01, 0.30, STUDIES), alpha=0.0).weights(_positions())
    assert np.allclose(flat, 1.0 / STUDIES)


def test_disagreed_drawn() -> None:
    drawn = _rising().drawn(_positions(), np.random.default_rng(7))
    assert len(drawn) == STUDIES
    assert drawn.dtype == np.int64
    assert (drawn >= STUDIES // 2).sum() > STUDIES // 2, "the disagreed half must be over-drawn"


def test_disagreed_tilt() -> None:
    assert _rising().tilt() == pytest.approx(0.5)


def test_disagreed_described() -> None:
    assert "0.50" in _rising().described()


def test_loaded(tmp_path: Path) -> None:
    weights = tmp_path / "spread.npz"
    np.savez(weights, studies=np.array(["b", "a"]), spread=np.array([0.30, 0.01]))
    policy = Disagreed.loaded(weights, ["a", "b"], alpha=0.5)
    assert policy.spread[0] == pytest.approx(0.01)
    assert policy.spread[1] == pytest.approx(0.30)


def test_loaded_unnamed_study_draws_at_the_mean(tmp_path: Path) -> None:
    weights = tmp_path / "spread.npz"
    np.savez(weights, studies=np.array(["a", "b"]), spread=np.array([0.10, 0.20]))
    policy = Disagreed.loaded(weights, ["a", "b", "c"], alpha=0.5)
    assert policy.spread[2] == pytest.approx(0.15)
