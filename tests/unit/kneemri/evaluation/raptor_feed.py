"""The raptor decode reimplementation, checked on the parts that are ours to get wrong.

The equality oracle in `raptor_feed.main` is the real proof and it needs their checkout and the corpus,
so it is not a unit test. These cover the pieces that carry a decision — the ordering, the span, the
slot fill, the window centres — on DICOM files written here.
"""

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from kneemri.evaluation.raptor_feed import (
    DEFAULT_SPACING_MM,
    EVAL_WINDOWS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    SLOTS,
    STACK_DEPTH,
    TRIPLET,
    RaptorFeed,
    Series,
    Slice,
    Windows,
)

SIDE = 8
"""Pixels a side in the written slices — small, because the decode resizes everything to `IMAGE_SIZE`."""

SPACING = 0.5


def write_series(directory: Path, count: int, *, spacing: float = SPACING) -> None:
    """Write `count` minimal single-frame DICOMs, deliberately in reverse anatomical order."""
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        slice_data = Dataset()
        slice_data.file_meta = FileMetaDataset()
        slice_data.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        slice_data.file_meta.MediaStorageSOPClassUID = CTImageStorage
        slice_data.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        slice_data.SOPClassUID = CTImageStorage
        slice_data.SOPInstanceUID = slice_data.file_meta.MediaStorageSOPInstanceUID
        slice_data.Rows, slice_data.Columns = SIDE, SIDE
        slice_data.SamplesPerPixel = 1
        slice_data.PhotometricInterpretation = "MONOCHROME2"
        slice_data.BitsAllocated, slice_data.BitsStored, slice_data.HighBit = 16, 16, 15
        slice_data.PixelRepresentation = 0
        slice_data.PixelSpacing = [spacing, spacing]
        slice_data.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        slice_data.ImagePositionPatient = [0, 0, float(count - index)]
        slice_data.InstanceNumber = count - index
        pixels = np.arange(SIDE * SIDE, dtype=np.uint16).reshape(SIDE, SIDE) * 4 + index * 10
        slice_data.PixelData = pixels.tobytes()
        slice_data.save_as(directory / f"{index:03d}.dcm", enforce_file_format=True)


def rows(uid: str) -> list[dict[str, object]]:
    """One series row per slot, so a study fills every slot from one directory tree."""
    return [{"SeriesInstanceUID": uid, "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": "1"}]


def test_read(tmp_path: Path) -> None:
    """Ordering is by anatomical position, not by the filename the files happen to carry."""
    write_series(tmp_path / "series", 4)
    series = Series.read(str(tmp_path / "series"))

    assert len(series.slices) == 4
    positions = [one.position for one in series.slices]
    assert positions == sorted(positions), "slices must come back ordered along the slice normal"
    assert positions[0] == 1.0, (
        "the file written LAST holds the lowest ImagePositionPatient, so it must sort FIRST — reading "
        "in filename order would put it fourth and mirror the whole stack"
    )
    assert series.median_spacing_mm == pytest.approx(SPACING)


def test_read_falls_back_when_a_header_carries_no_geometry(tmp_path: Path) -> None:
    """A file that is not DICOM at all is ordered first rather than losing the whole series."""
    write_series(tmp_path / "series", 2)
    (tmp_path / "series" / "junk.dcm").write_bytes(b"not a dicom file")
    series = Series.read(str(tmp_path / "series"))

    assert len(series.slices) == TRIPLET
    assert series.slices[0].position == 0.0
    assert series.slices[0].spacing_mm == pytest.approx(DEFAULT_SPACING_MM)


def test_sampled() -> None:
    """The span is 6–94%, which is the whole reason the peripheral findings survive."""
    series = Series(tuple(Slice(float(i), f"{i}.dcm", SPACING) for i in range(100)), SPACING)
    picked = series.sampled(5)

    assert [one.path for one in picked] == ["6.dcm", "28.dcm", "50.dcm", "71.dcm", "93.dcm"], (
        "int(100*0.06)=6 and int(100*0.94)-1=93 are the endpoints; a 0.15-0.85 band would start at 15 "
        "and cut the collateral ligaments the model is scored on"
    )


def test_sampled_repeats_a_single_slice_rather_than_leaving_the_slot_short() -> None:
    """One slice still has to fill its slot, or the depths behind it shift."""
    series = Series((Slice(0.0, "only.dcm", SPACING),), SPACING)

    assert [one.path for one in series.sampled(TRIPLET)] == ["only.dcm"] * TRIPLET


def test_records(tmp_path: Path) -> None:
    """Every value stays a `str`, because the slot match compares `Fluid_Sensitive` against `"1"`."""
    csv_path = tmp_path / "train_series.csv"
    csv_path.write_text(
        "StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane,Fluid_Sensitive\n"
        "study-a,series-a,Sagittal,1\nstudy-a,series-b,Coronal,0\nstudy-b,series-c,Axial,1\n",
        encoding="utf-8",
    )

    records = RaptorFeed.records(csv_path)

    assert set(records) == {"study-a", "study-b"}
    assert len(records["study-a"]) == 2
    assert records["study-a"][0]["Fluid_Sensitive"] == "1", (
        "read as an int, the slot match would silently never fire and every study would decode empty"
    )


def test_study(tmp_path: Path) -> None:
    """One sagittal series fills its own slot and leaves every later slot zeroed and masked out."""
    uid = "series-a"
    write_series(tmp_path / "study-a" / uid, 6)
    volume, mask = RaptorFeed(tmp_path).study("study-a", {"study-a": rows(uid)})

    assert volume.shape == (STACK_DEPTH, IMAGE_SIZE, IMAGE_SIZE)
    filled = SLOTS[0].slices
    assert mask.sum() == filled, "only the fluid-sensitive sagittal slot has a series to fill it"
    assert mask[:filled].all() and not mask[filled:].any(), (
        "the filled depths must be the slot's own, not scattered — a shifted fill is the failure that "
        "still produces a well-formed volume and scores as a merely weak member"
    )
    assert volume[filled:].sum() == 0


def test_study_leaves_the_volume_empty_when_no_series_matches(tmp_path: Path) -> None:
    """A study with no usable plane is all zeros and says so in the mask."""
    volume, mask = RaptorFeed(tmp_path).study("absent", {})

    assert volume.sum() == 0
    assert mask.sum() == 0


def test_centres() -> None:
    """Every centre sits strictly inside the filled span, so a window never runs off the stack."""
    mask = np.zeros(STACK_DEPTH, np.uint8)
    mask[4:40] = 1
    centres = Windows().centres(mask)

    assert len(centres) == EVAL_WINDOWS
    assert min(centres) > 4 and max(centres) < 39, "a centre on the edge would stack a zeroed neighbour"
    assert centres == sorted(centres)


def test_centres_survives_a_mask_with_almost_nothing_in_it() -> None:
    """A one-slice study still yields centres rather than raising on an empty range."""
    mask = np.zeros(STACK_DEPTH, np.uint8)
    mask[7] = 1

    assert len(Windows().centres(mask)) == EVAL_WINDOWS


def test_stacked() -> None:
    """A window is the slice and its neighbours, normalised the way the weights expect."""
    volume = np.zeros((STACK_DEPTH, SIDE, SIDE), np.uint8)
    volume[2:10] = 255
    mask = (volume.reshape(STACK_DEPTH, -1).sum(1) > 0).astype(np.uint8)
    windows = Windows(count=4).stacked(volume, mask, SIDE)

    assert windows.shape == (4, TRIPLET, SIDE, SIDE)
    interior = float(windows[0, 0].min())
    assert interior == pytest.approx(float((1.0 - IMAGENET_MEAN[0, 0, 0]) / 0.229), rel=1e-4), (
        "a saturated pixel must land at (1 - mean) / std; a missing normalisation reads as 1.0 and "
        "shifts every activation the checkpoint was fitted under"
    )


def test_stacked_resizes_when_the_model_asks_for_another_resolution() -> None:
    """The window count is fixed by the mask; the resolution is whatever the arm was trained at."""
    volume = np.zeros((STACK_DEPTH, SIDE, SIDE), np.uint8)
    volume[2:10] = 200
    mask = (volume.reshape(STACK_DEPTH, -1).sum(1) > 0).astype(np.uint8)

    assert Windows(count=2).stacked(volume, mask, SIDE * 2).shape == (2, TRIPLET, SIDE * 2, SIDE * 2)


def test_read_px_inverts_a_monochrome1_study(tmp_path: Path) -> None:
    """MONOCHROME1 stores white-on-black, and a study read uninverted is a negative of itself."""
    write_series(tmp_path / "series", 1)
    path = next((tmp_path / "series").glob("*.dcm"))
    slice_data = pydicom.dcmread(path)
    slice_data.PhotometricInterpretation = "MONOCHROME1"
    slice_data.save_as(path, enforce_file_format=True)

    pixels = RaptorFeed._pixels(str(path))

    assert pixels is not None
    assert pixels.max() == pytest.approx((SIDE * SIDE - 1) * 4), (
        "the darkest written pixel is 0 and must come back as the brightest; left uninverted the study "
        "is a photographic negative of itself and every finding is scored on the wrong contrast"
    )
