from pathlib import Path

import torch

from core.data.corpus.acquisition import Acquisition
from core.data.corpus.laterality import Laterality
from tests.unit.conftest import write_series


def _acquisition(tmp_path: Path, name: str, **parts: object) -> Acquisition:
    return Acquisition.probe("study", name, write_series(tmp_path / name, **parts))  # pyrefly: ignore[bad-argument-type]


def test_of_study(tmp_path: Path):
    """The tag decides where it exists; geometry decides where it does not; the midline stays unresolved."""
    tagged = _acquisition(tmp_path, "tagged", laterality="R")
    assert Laterality().of_study([tagged]) == "R"

    # Untagged, centre on the patient's right (negative x). The synthetic centre sits 9 mm from the
    # midline, inside the default 20 mm dead zone, so this asserts BOTH halves of the rule: resolved
    # when the dead zone admits it, and deliberately unresolved when it does not.
    right = _acquisition(tmp_path, "right", laterality=None, normal=(-1.0, 0.0, 0.0))
    assert Laterality(dead_zone_mm=5.0).of_study([right]) == "R"
    assert Laterality().of_study([right]) is None

    near_midline = _acquisition(tmp_path, "midline", laterality=None, spacing=0.01, size=8)
    assert Laterality(dead_zone_mm=1e6).of_study([near_midline]) is None, "inside the dead zone, no guess"


def test_of_study_returns_none_without_tag_or_geometry(tmp_path: Path):
    """A study with neither is unresolved, which means simply not mirrored — never assumed left."""
    assert Laterality().of_study([]) is None


def test_normalise():
    """Coronal and axial mirror in-plane; sagittal reverses the SLICE order instead."""
    images = torch.arange(2 * 2 * 3, dtype=torch.uint8).reshape(2, 2, 3)
    assert torch.equal(Laterality().normalise(images, "Coronal", "R"), torch.flip(images, dims=[-1]))
    assert torch.equal(Laterality().normalise(images, "Sagittal", "R"), torch.flip(images, dims=[0]))


def test_normalise_leaves_left_and_unresolved_alone():
    """Only a right knee is mapped; a left one and an unresolved one pass through untouched."""
    images = torch.arange(6, dtype=torch.uint8).reshape(1, 2, 3)
    assert torch.equal(Laterality().normalise(images, "Sagittal", "L"), images)
    assert torch.equal(Laterality().normalise(images, "Sagittal", None), images)
