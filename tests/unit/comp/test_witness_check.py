"""The residual's SHAPE, which is what separates a rounding from a reading defect (knee-djk8)."""

import numpy as np
import torch

from comp.witness_check import EPS_FACTOR, spread, tolerance


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


def test_tolerance_admits_a_bfloat16_member_that_drifts_by_one_eps() -> None:
    """THE EQUIVALENCE CLASS THAT LET THE BUG LIVE: a bf16 member whose residual is one of its own eps.

    The gate used to hand the cuda path a flat 2e-3, which is two float16 eps and a QUARTER of one
    bfloat16 eps, so this member — drifting by a single rounding — was refused for its dtype rather
    than for its reading. Written in eps rather than in the number it evaluates to, because a bound
    that is one arithmetic step wide is the claim; 7.8e-3 is only today's spelling of it.
    """
    one_eps = float(torch.finfo(torch.bfloat16).eps)
    assert tolerance(torch.bfloat16) > one_eps


def test_tolerance_refuses_the_reading_defects_the_gate_exists_for() -> None:
    """The loosening is only acceptable while it still discriminates: those four moved 0.05-0.5."""
    smallest_real_defect = 0.05
    assert all(tolerance(dtype) < smallest_real_defect for dtype in (torch.float16, torch.bfloat16))


def test_tolerance_scales_with_the_mantissa_rather_than_being_picked() -> None:
    """bfloat16 spends seven bits where float16 spends ten, so its bound is eight times as wide."""
    assert tolerance(torch.bfloat16) == EPS_FACTOR * float(torch.finfo(torch.bfloat16).eps)
    assert tolerance(torch.bfloat16) > tolerance(torch.float16)
