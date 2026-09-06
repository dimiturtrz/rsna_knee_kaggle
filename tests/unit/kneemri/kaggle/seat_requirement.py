"""Unit tests for the seat-requirement inversion.

Everything here is exercised on synthetic readings with a KNOWN plane, so a wrong coefficient shows up
as a wrong recovered plane rather than as a plausible-looking number nobody can check.
"""

from pathlib import Path

from kneemri.kaggle.seat_requirement import SeatFit, SeatReading, SeatRequirement

PLANE_INTERCEPT = -0.5
PER_SOLO = 0.4
PER_AGREEMENT = -0.2
TOLERANCE = 1e-9


def on_the_plane(solo: float, agreement: float) -> SeatReading:
    gain = PLANE_INTERCEPT + PER_SOLO * solo + PER_AGREEMENT * agreement
    return SeatReading(name=f"m_{solo}_{agreement}", solo=solo, agreement=agreement, gain=gain)


def a_fit() -> SeatFit:
    return SeatFit(
        intercept=PLANE_INTERCEPT,
        per_solo=PER_SOLO,
        per_agreement=PER_AGREEMENT,
        residual=0.0,
        best_observed=0.002,
    )


def test_gain() -> None:
    expected = PLANE_INTERCEPT + PER_SOLO * 0.86 + PER_AGREEMENT * 0.9
    assert abs(a_fit().gain(0.86, 0.9) - expected) < TOLERANCE


def test_solo_needed() -> None:
    fit = a_fit()
    needed = fit.solo_needed(agreement=0.9, target=0.01)
    assert abs(fit.gain(needed, 0.9) - 0.01) < TOLERANCE
    assert SeatFit(0.0, 0.0, PER_AGREEMENT, 0.0, 0.002).solo_needed(0.9, 0.01) == float("inf")


def test_agreement_needed() -> None:
    fit = a_fit()
    needed = fit.agreement_needed(solo=0.86, target=0.01)
    assert abs(fit.gain(0.86, needed) - 0.01) < TOLERANCE
    assert SeatFit(0.0, PER_SOLO, 0.0, 0.0, 0.002).agreement_needed(0.86, 0.01) == float("inf")


def test_extrapolation() -> None:
    assert abs(a_fit().extrapolation(0.01) - 5.0) < TOLERANCE
    assert SeatFit(0.0, PER_SOLO, PER_AGREEMENT, 0.0, 0.0).extrapolation(0.01) == float("inf")


def test_fitted() -> None:
    corners = ((0.80, 0.90), (0.86, 0.92), (0.84, 0.85), (0.88, 0.95))
    readings = [on_the_plane(solo, agreement) for solo, agreement in corners]
    fit = SeatRequirement.fitted(readings)
    assert fit is not None
    assert abs(fit.intercept - PLANE_INTERCEPT) < 1e-6
    assert abs(fit.per_solo - PER_SOLO) < 1e-6
    assert abs(fit.per_agreement - PER_AGREEMENT) < 1e-6
    assert fit.residual < 1e-9
    assert SeatRequirement.fitted(readings[:2]) is None


def test_report(tmp_path: Path) -> None:
    corners = ((0.80, 0.90), (0.86, 0.92), (0.84, 0.85), (0.88, 0.95))
    readings = [on_the_plane(solo, agreement) for solo, agreement in corners]
    lines = SeatRequirement(tmp_path).report(target=0.01, readings=readings)
    assert any(line.startswith("fit: gain =") for line in lines)
    assert any("solo must reach" in line for line in lines)
    assert any("agreement must fall to" in line for line in lines)
    assert SeatRequirement(tmp_path).report(target=0.01, readings=readings[:2])[0].startswith("REFUSED")
