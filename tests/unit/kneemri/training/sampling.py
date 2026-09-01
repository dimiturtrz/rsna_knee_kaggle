"""The mirror for the epoch's draw policy: every public method called, and asserted."""

from pathlib import Path

import numpy as np
import pytest

from kneemri.training.sampling import Disagreed, Ramped, Uniform

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


def test_disagreed_loaded(tmp_path: Path) -> None:
    weights = tmp_path / "spread.npz"
    np.savez(weights, studies=np.array(["b", "a"]), spread=np.array([0.30, 0.01]))
    policy = Disagreed.loaded(weights, ["a", "b"], alpha=0.5)
    assert policy.spread[0] == pytest.approx(0.01)
    assert policy.spread[1] == pytest.approx(0.30)


def test_disagreed_loaded_unnamed_study_draws_at_the_mean(tmp_path: Path) -> None:
    weights = tmp_path / "spread.npz"
    np.savez(weights, studies=np.array(["a", "b"]), spread=np.array([0.10, 0.20]))
    policy = Disagreed.loaded(weights, ["a", "b", "c"], alpha=0.5)
    assert policy.spread[2] == pytest.approx(0.15)


def _ramped(epochs: int = 4) -> Ramped:
    """The same rising disagreement, but reached over `epochs` epochs rather than at the first step."""
    return Ramped(np.linspace(0.01, 0.30, STUDIES), alpha=0.5, epochs=epochs)


def test_ramped_drawn() -> None:
    """The tilt arrives over the ramp: epoch zero is the flat draw, the last epoch is the full tilt."""
    positions = _positions()
    generator = np.random.default_rng(0)
    policy = _ramped()

    alphas = []
    for _ in range(6):
        drawn = policy.drawn(positions, generator)
        alphas.append(policy.alpha)
        assert drawn.shape == positions.shape
        assert drawn.dtype == np.int64

    assert alphas[0] == pytest.approx(0.0), "the first epoch must be the alpha-zero control's draw"
    assert alphas == sorted(alphas), "the tilt may never go backwards"
    assert alphas[3] == pytest.approx(0.375)
    assert alphas[4] == pytest.approx(0.5), "the ramp reaches its destination on schedule"
    assert alphas[5] == pytest.approx(0.5), "and holds there rather than overshooting"


def test_ramped_drawn_reaches_the_static_policy() -> None:
    """Past the ramp this IS `Disagreed` at the same alpha — the two differ only in the approach."""
    positions = _positions()
    policy = _ramped(epochs=1)
    policy.drawn(positions, np.random.default_rng(0))

    ramped = policy.drawn(positions, np.random.default_rng(7))
    static = Disagreed(np.linspace(0.01, 0.30, STUDIES), alpha=0.5).drawn(positions, np.random.default_rng(7))

    assert np.array_equal(ramped, static)


def test_ramped_tilt() -> None:
    """The DESTINATION, not the current value — `landed` reads this before the first epoch."""
    policy = _ramped()
    assert policy.tilt() == pytest.approx(0.5)
    assert policy.alpha == pytest.approx(0.0), "and the ramp really has not started yet"


def test_ramped_described() -> None:
    assert "ramping to alpha 0.50 over 4 epochs" in _ramped().described()


def test_ramped_loaded(tmp_path: Path) -> None:
    weights = tmp_path / "spread.npz"
    np.savez(weights, studies=np.array(["b", "a"]), spread=np.array([0.30, 0.01]))

    policy = Ramped.loaded(weights, ["a", "b"], alpha=0.5, epochs=3)

    assert policy.spread[0] == pytest.approx(0.01), "the identifier join is the parent's and must survive"
    assert policy.epochs == 3
    assert policy.tilt() == pytest.approx(0.5)


def test_sampling_ramp() -> None:
    """Every policy with no schedule answers zero, which is what makes the field readable at all.

    `landed` compares a recorded ramp against the flag that asked for one, so a policy that simply
    has no ramp must answer a NUMBER rather than raise — the alternative is a switch check that can
    only run on the one policy that carries a schedule.
    """
    assert Uniform().ramp() == 0
    assert Disagreed(np.linspace(0.01, 0.30, STUDIES), alpha=0.5).ramp() == 0


def test_ramped_ramp() -> None:
    """The schedule length, which `tilt` cannot say: both arms of the 2x2 report the same tilt."""
    assert _ramped(epochs=6).ramp() == 6
    assert _ramped(epochs=6).tilt() == pytest.approx(0.5), "and the tilt is the destination either way"
