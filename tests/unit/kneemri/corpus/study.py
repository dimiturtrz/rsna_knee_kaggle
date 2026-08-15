from pathlib import Path

import torch

from kneemri.corpus.slot_reader import SlotReader
from kneemri.corpus.slots import EXTENDED_SLOTS, SCHEMES, SLOTS, SlotScheme
from kneemri.corpus.study import SlotPipeline, Study

from ..conftest import write_series


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
    left = pipeline.slot_images(acquisition, 3, SLOTS[0], "L")
    right = pipeline.slot_images(acquisition, 3, SLOTS[0], "R")
    assert left is not None and right is not None
    assert torch.equal(right, torch.flip(left, dims=[0])), "a right knee reverses the sagittal stack"


def test_bag_widths_to_the_pipeline_scheme(tmp_path: Path):
    """A nine-slot scheme produces nine bags, because `route` already routes nine.

    The two halves of `bag` used to disagree: routing honoured the pipeline's scheme while the emit
    loop iterated the module's canonical six. The cache is allocated at the scheme's width, so an
    extended build would have raised a shape error on EVERY study, been caught as "decode failed", and
    left a fully zero cache whose mask claims nothing was acquired — a build that finishes and a
    trainer that reads zeros.
    """
    root, planes = _study(tmp_path)
    extended = SlotPipeline(scheme=SlotScheme(EXTENDED_SLOTS), reader=SlotReader(image_size=16))
    images, mask = Study.from_directory(root, planes, extended).bag(slices=3)
    assert images.shape == (len(EXTENDED_SLOTS), 3, 16, 16)
    assert len(mask) == len(EXTENDED_SLOTS)
    assert len(EXTENDED_SLOTS) > len(SLOTS), "the point of the scheme is that it is wider"


def test_bags(tmp_path: Path):
    """Every requested size comes back from ONE read, each identical to a native read at that size.

    Identity is the whole point. `predict_groups` decodes the split once per pixel group, so a
    submission carrying members at 392, 336 and 224 px reads every DICOM three times — and the only
    version of that fix worth having is one where no member's pixels change, because a member's
    fingerprint and witness were recorded under a native read.
    """
    root, planes = _study(tmp_path)
    study = Study.from_directory(root, planes, SlotPipeline(reader=SlotReader(image_size=32)))

    bags = study.bags(3, (32, 16))

    assert sorted(bags) == [16, 32]
    for size in (32, 16):
        images, mask = bags[size]
        native, native_mask = Study.from_directory(root, planes, SlotPipeline(reader=SlotReader(image_size=size))).bag(
            3
        )
        assert torch.equal(images, native), f"one read must equal a native read at {size}"
        assert torch.equal(mask, native_mask), "the mask is a property of the study, not of the grid"


def test_slot_sizes(tmp_path: Path):
    """A slot read once and delivered at every size, or None when the series will not decode."""
    root, planes = _study(tmp_path)
    pipeline = SlotPipeline(reader=SlotReader(image_size=32))
    study = Study.from_directory(root, planes, pipeline)
    acquisition = study.acquisitions[0]

    sized = pipeline.slot_sizes(acquisition, 3, SLOTS[0], "L", (32, 16))

    assert sized is not None
    assert sorted(sized) == [16, 32]
    assert sized[16].shape == (3, 16, 16)
    assert sized[32].dtype == torch.uint8


def test_bags_reads_a_multiscale_scheme_at_two_extents(tmp_path: Path):
    """Twelve slots over six sequences: the same series wide and tight, from ONE decode.

    The canonical six must be byte-identical to what a canonical scheme produces, because every cache on
    disk and every member fitted on one depends on those first six positions meaning what they meant.
    """
    # 3 mm spacing so BOTH extents fit a 64-pixel image: 130 mm is 43 px and 80 mm is 27 px. At the
    # default 0.3 mm neither crop applies (130 mm wants 433 px) and the two scales come back identical,
    # which is the fixture failing to exercise the thing rather than the thing not working.
    write_series(tmp_path / "sag", slices=4, description="SAG PD FS", size=64, spacing=3.0)
    planes = {"sag": "Sagittal"}
    study = Study.from_directory(tmp_path, planes, SlotPipeline(reader=SlotReader(image_size=16)))
    wide = study.bags(2, (16,))[16]

    multi = Study.from_directory(
        tmp_path, planes, SlotPipeline(scheme=SlotScheme(SCHEMES["multiscale"]), reader=SlotReader(image_size=16))
    ).bags(2, (16,))[16]

    assert multi[0].shape[0] == 12, "six sequences read at two extents"
    assert torch.equal(multi[0][:6], wide[0]), "the canonical six are unchanged, byte for byte"
    assert multi[1][0] == 1.0 and multi[1][6] == 1.0, "a filled slot is filled at both scales"
    assert not torch.equal(multi[0][6], multi[0][0]), "and the tight read is different pixels"
