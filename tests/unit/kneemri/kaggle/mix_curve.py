"""The mix curve is arithmetic over two measured endpoints, so every test states the arithmetic."""

import math

import numpy as np

from kneemri.kaggle.mix_curve import FOREIGN_SERVED, MixCurve, SeatCurve

FLAT = SeatCurve(base=0.80, solo=0.90, blends=(0.85, 0.90), members=1)
"""One member, a candidate at one then two copies: weights 1/2 and 2/3, and both blends sit ON the line."""


def test_weights() -> None:
    assert np.allclose(FLAT.weights, [0.5, 2.0 / 3.0])


def test_linear() -> None:
    assert np.allclose(FLAT.linear(), [0.85, 0.80 / 3.0 + 0.60])


def test_synergy() -> None:
    assert math.isclose(FLAT.synergy()[0], 0.0, abs_tol=1e-12)
    assert np.allclose(FOREIGN_SERVED.synergy() > 0.0, True)


def test_synergy_at() -> None:
    assert math.isclose(FLAT.synergy_at(0.5), 0.0, abs_tol=1e-12)
    assert math.isclose(FLAT.synergy_at(0.01), FLAT.synergy()[0], abs_tol=1e-12)


def test_board_at() -> None:
    curve = MixCurve(FLAT, board_base=0.94)
    assert math.isclose(curve.board_at(0.90, 0.5, synergy_share=0.0), 0.92, abs_tol=1e-12)


def test_implied_solo() -> None:
    curve = MixCurve(FLAT, board_base=0.94)
    assert math.isclose(curve.implied_solo(0.92, 0.5, synergy_share=0.0), 0.90, abs_tol=1e-12)


def test_transferred() -> None:
    curve = MixCurve(FOREIGN_SERVED)
    weight = 0.4615
    bump = FOREIGN_SERVED.synergy_at(weight)
    whole = curve.board_at(0.90, weight, synergy_share=1.0)
    assert math.isclose(curve.transferred(0.90, whole, weight), 1.0, abs_tol=1e-9)
    assert math.isclose(curve.transferred(0.90, whole - bump, weight), 0.0, abs_tol=1e-9)


def test_best_weight() -> None:
    """A candidate below the base with no synergy earns no seat at all."""
    weight, score = MixCurve(FLAT, board_base=0.94).best_weight(0.80, synergy_share=0.0)
    assert weight == 0.0
    assert math.isclose(score, 0.94, abs_tol=1e-12)


def test_lines() -> None:
    lines = MixCurve(FOREIGN_SERVED).lines(0.914)
    assert any("the board paid" in line for line in lines)
    assert len(lines) == len(FOREIGN_SERVED.blends) + 7
