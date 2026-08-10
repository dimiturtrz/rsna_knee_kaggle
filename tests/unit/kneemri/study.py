from pathlib import Path

import torch

from kneemri.slot_reader import SlotReader
from kneemri.slots import SLOTS
from kneemri.study import SlotPipeline, Study

from .conftest import write_series


def _study(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A study of two series: one sagittal fluid fat-suppressed, one coronal T1."""
    root = tmp_path / "study"
    write_series(root / "sag", description="SAG PD FS")
    write_series(root / "cor", description="COR T1", repetition=500.0, echo=10.0, scan_options="")
    return root, {"sag": "Sagittal", "cor": "Coronal"}


def test_from_directory(tmp_path: Path):
    """Every series directory under the study is probed, one header each."""
    root, planes = _study(tmp_path)
    study = Study.from_directory(root, planes)
    assert {acquisition.series for acquisition in study.acquisitions} == {"sag", "cor"}


def test_side(tmp_path: Path):
    """The study's side comes from the tag when the series carry one."""
    root, planes = _study(tmp_path)
    assert Study.from_directory(root, planes).side() == "L"


def test_bag(tmp_path: Path):
    """The bag is one stack per slot in the fixed order, with the mask marking what was filled."""
    root, planes = _study(tmp_path)
    study = Study.from_directory(root, planes, SlotPipeline(reader=SlotReader(image_size=16)))
    images, mask = study.bag(slices=3)
    assert images.shape == (len(SLOTS), 3, 16, 16)
    assert images.dtype == torch.uint8
    assert mask.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], "sagittal fluid-FS and coronal T1 only"
    assert int(images[1].max()) == 0, "an unfilled slot is zeros, and the mask says so"


def test_image_size(tmp_path: Path):
    """The pipeline reports the resolution it will produce, so callers size their zeros from it."""
    assert SlotPipeline(reader=SlotReader(image_size=16)).image_size == 16


def test_side_of(tmp_path: Path):
    """The pipeline answers which knee it is, rather than exposing its laterality rule to be reached through."""
    root, _ = _study(tmp_path)
    study = Study.from_directory(root, {})
    assert SlotPipeline().side_of(study.acquisitions) == "L"


def test_route(tmp_path: Path):
    """Routing is the pipeline's job, under whatever slot predicate it was configured with."""
    root, planes = _study(tmp_path)
    study = Study.from_directory(root, planes)
    chosen = SlotPipeline().route(study.acquisitions, planes)
    assert set(chosen) == {"SAG_FLUID_FS", "COR_T1"}


def test_slot_images(tmp_path: Path):
    """Reading and laterality normalisation happen together — they are never useful apart."""
    root, planes = _study(tmp_path)
    study = Study.from_directory(root, planes)
    pipeline = SlotPipeline(reader=SlotReader(image_size=16))
    acquisition = pipeline.route(study.acquisitions, planes)["SAG_FLUID_FS"]
    left = pipeline.slot_images(acquisition, 3, "Sagittal", "L")
    right = pipeline.slot_images(acquisition, 3, "Sagittal", "R")
    assert left is not None and right is not None
    assert torch.equal(right, torch.flip(left, dims=[0])), "a right knee reverses the sagittal stack"
