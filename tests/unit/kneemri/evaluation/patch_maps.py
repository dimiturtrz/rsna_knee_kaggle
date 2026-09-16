"""The three readings of a patch sweep, each against a map whose right answer is known by construction."""

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.patch_maps import CENTRE_SHARE_NULL, FINEST_GRID, PatchMaps

STUDIES = 4
GRID = 4
PEAK = 1.0
FLOOR = 0.1


def _peaked(row: int = 1, column: int = 2) -> PatchMaps:
    """Every study and label agreeing on one cell, over a floor of small values everywhere else."""
    maps = np.full((STUDIES, len(TARGETS), GRID, GRID), FLOOR, dtype=np.float32)
    maps[:, :, row, column] = PEAK
    return PatchMaps(maps)


def _diffuse() -> PatchMaps:
    """The null: every cell worth exactly the same, so no centre and no place."""
    return PatchMaps(np.full((STUDIES, len(TARGETS), GRID, GRID), FLOOR, dtype=np.float32))


def test_centre_mass():
    """A flat map holds the null share in the centre; a map peaked inside it holds more."""
    assert _diffuse().centre_mass()["ACL"] == pytest.approx(CENTRE_SHARE_NULL)
    assert _peaked().centre_mass()["ACL"] > CENTRE_SHARE_NULL


def test_centre_mass_refuses_a_grid_with_no_centre():
    """Two cells have no central half, and the degenerate share is 1.000 — a pass that was never asked.

    Measured 2026-09-16 on a `--grid 2` smoke: every one of the twelve labels read `centre 1.000` and
    the report said twelve of twelve were above the diffuse null.
    """
    coarse = np.full((STUDIES, len(TARGETS), FINEST_GRID // 2, FINEST_GRID // 2), FLOOR, dtype=np.float32)

    with pytest.raises(ValueError, match="no central half"):
        PatchMaps(coarse).centre_mass()


def test_centroids():
    """The centroid lands on the peaked cell, and studies that agree report no disagreement."""
    row, column, spread = _peaked(row=1, column=2).centroids()["ACL"]

    assert row == pytest.approx(1.0, abs=0.4), "the floor pulls the centroid toward the middle, it does not move it"
    assert column == pytest.approx(2.0, abs=0.4)
    assert spread == pytest.approx(0.0, abs=1e-6), "identical studies cannot disagree about a centroid"


def test_label_contrast():
    """Positives peaked on one cell and negatives on another must separate by the distance between them.

    This is the reading that survives the centre-mass confound: a map centred because the middle of a
    knee holds more tissue looks IDENTICAL for a positive and a negative study, so any separation here
    came from the label and nowhere else.
    """
    maps = np.full((STUDIES, len(TARGETS), GRID, GRID), FLOOR, dtype=np.float32)
    truth = np.zeros((STUDIES, len(TARGETS)), dtype=np.float32)
    positive = np.arange(STUDIES) % 2 == 0
    truth[positive] = 1.0
    maps[positive, :, 0, 0] = PEAK
    maps[~positive, :, 3, 3] = PEAK

    separations = PatchMaps(maps, truth).label_contrast()

    assert separations["ACL"] > 1.0, "two corners cannot report the same centroid"
    assert np.isnan(PatchMaps(maps, np.ones_like(truth)).label_contrast()["ACL"]), "no negatives, no contrast"


def test_lines():
    """The report names every label and calls the flat map what it is."""
    lines = _diffuse().lines()

    assert any(label in line for label in TARGETS for line in lines)
    assert any("DIFFUSE" in line for line in lines)
    assert f"0 of {len(TARGETS)} labels above the diffuse null" in lines[-2]
