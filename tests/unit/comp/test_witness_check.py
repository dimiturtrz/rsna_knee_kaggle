"""The residual's SHAPE, which is what separates a rounding from a reading defect (knee-djk8)."""

import numpy as np

from comp.witness_check import spread


def test_spread_reports_every_label_when_the_residual_is_uniform() -> None:
    """A rounding touches all twelve columns about equally, because they come off the same GEMMs."""
    residual = np.full((8, 12), 1e-3)
    assert spread(residual) == "12 of 12 labels carry it"


def test_spread_reports_one_label_when_the_residual_concentrates() -> None:
    """The four defects this check exists for moved a FEW findings by 0.05-0.5 and spared the rest."""
    residual = np.full((8, 12), 1e-4)
    residual[:, 3] = 0.2
    assert spread(residual) == "1 of 12 labels carry it"


def test_spread_says_exact_rather_than_dividing_by_zero() -> None:
    """Three members reproduce their witness bit-for-bit; the widest label is then 0."""
    assert spread(np.zeros((8, 12))) == "exact in every label"
