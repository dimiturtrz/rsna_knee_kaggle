"""fastMRI+ boxes through the flip, the band, the window and the resize, on volumes the test writes itself."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from core.data.corpus.fastmri_boxes import BOX_TABLE, STUDY_TABLE, BoxFrame, FastmriBoxes
from core.data.corpus.fastmri_corpus import RECONSTRUCTION, FastmriVolume
from core.data.corpus.slots import FASTMRI_SCHEME, Routing
from core.data.corpus.study_cache import StudyCache

DEPTH, MATRIX, SIZE, SLICES = 10, 32, 8, 4
CROPPING = 130.0 / 24.0
"""A spacing at which the default 130 mm crop takes 24 of 32 pixels, centred: rows and columns 4..28."""


def _annotations(tmp_path: Path) -> Path:
    """Two boxed studies, one study-level row with no geometry, and a read study with no box at all.

    The file list is written headerless with its trailing `Knee data` footer, as the corpus ships it.
    """
    path = tmp_path / "Annotations"
    path.mkdir()
    pd.DataFrame(
        [
            ["file100", 3, "No", 8.0, 20.0, 4.0, 6.0, "Meniscus Tear"],
            ["file100", 9, "No", 0.0, 0.0, 2.0, 2.0, "Joint Bodies"],
            ["file101", 5, "No", 12.0, 2.0, 8.0, 4.0, "Ligament - ACL High Grade Sprain"],
            ["file101", np.nan, "Yes", np.nan, np.nan, np.nan, np.nan, "artifact"],
        ],
        columns=["file", "slice", "study_level", "x", "y", "width", "height", "label"],
    ).to_csv(path / "knee.csv", index=False)
    listed = pd.DataFrame({0: ["file100", "file101", "file102", "Knee data"]})
    listed.to_csv(path / "knee_file_list.csv", index=False, header=False)
    return path


def _volumes(tmp_path: Path) -> list[FastmriVolume]:
    """Three reconstructions on disk, plus one never read, which must get no row."""
    volumes = []
    for index, name in enumerate(("file100", "file101", "file102", "file103")):
        path = tmp_path / f"{name}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset(RECONSTRUCTION, data=np.random.default_rng(index).random((DEPTH, MATRIX, MATRIX)))
        volumes.append(FastmriVolume(name, path, 0, MATRIX, CROPPING * MATRIX))
    return volumes


def _boxes(tmp_path: Path) -> FastmriBoxes:
    cache = StudyCache(tmp_path, slices=SLICES, image_size=SIZE, routing=Routing(scheme=FASTMRI_SCHEME))
    return FastmriBoxes(_annotations(tmp_path), cache)


def test_place():
    """Corners scale about the window origin, clip to the frame, and a box cut away reads invisible."""
    frame = BoxFrame(kept=(1, 3, 5, 7), top=4, left=4, rows=24, columns=24, size=SIZE)
    native = pd.DataFrame(
        {
            "slice": [3, 9],
            "native_x0": [10.0, 0.0],
            "native_y0": [7.0, 0.0],
            "native_x1": [16.0, 2.0],
            "native_y1": [31.0, 3.0],
        }
    )

    placed = frame.place(native)

    assert placed["cache_slice"].tolist() == [1, 3], "the nearest KEPT slice, as a position in the cache"
    assert placed["slice_gap"].tolist() == [0, 2], "a box off the band is marked by its distance, not dropped"
    assert placed.loc[0, ["cache_x0", "cache_y0", "cache_x1", "cache_y1"]].tolist() == pytest.approx(
        [2.0, 1.0, 4.0, SIZE]
    ), "(10-4)/3, (7-4)/3, (16-4)/3, and (31-4)/3 clipped to the frame"
    assert placed["visible"].tolist() == [True, False], "wholly outside the window"
    assert placed["slice"].tolist() == [3, 9], "the original slice index survives"


def test_native():
    """The drawn image was flipped up/down: a box drawn at the top sits at the bottom of the volume, x unmoved."""
    drawn = pd.DataFrame({"x": [8.0], "y": [2.0], "width": [4.0], "height": [6.0]})

    native = FastmriBoxes.native(drawn, rows=MATRIX)

    assert native.loc[0, ["native_x0", "native_y0", "native_x1", "native_y1"]].tolist() == [8.0, 24.0, 12.0, 30.0]


def test_frame(tmp_path: Path):
    """The frame is the one the cache reader takes: its band, its window, its size."""
    boxes = _boxes(tmp_path)
    pixels = np.random.default_rng(0).random((DEPTH, MATRIX, MATRIX)).astype(np.float32)

    cropped = boxes.frame(pixels, CROPPING)
    whole = boxes.frame(pixels, 0.4375)

    assert cropped.kept == tuple(boxes.cache.reader().band_indices(DEPTH, SLICES))
    assert (cropped.top, cropped.left, cropped.rows, cropped.columns, cropped.size) == (4, 4, 24, 24, SIZE)
    assert (whole.top, whole.rows) == (0, MATRIX), "a crop wider than the frame is no crop"


def test_tables(tmp_path: Path):
    """Every geometric box of a read study, and every read study on disk, a clean one kept at zero boxes."""
    boxes, studies = _boxes(tmp_path).tables(_volumes(tmp_path))

    assert len(boxes) == 3
    assert studies["file"].tolist() == ["file100", "file101", "file102"], "file103 was never read"
    assert studies["boxes"].tolist() == [2, 1, 0], "the negative keeps its row"
    acl = boxes[boxes["file"] == "file101"].iloc[0]
    assert (acl["native_y0"], acl["native_y1"]) == (26.0, 30.0)


def test_write(tmp_path: Path):
    """Both tables land under names carrying the cache key."""
    boxes = _boxes(tmp_path)

    boxes.write(_volumes(tmp_path), tmp_path / "labels")

    key = boxes.cache.key
    assert len(pd.read_csv(tmp_path / "labels" / BOX_TABLE.format(key=key))) == 3
    assert len(pd.read_csv(tmp_path / "labels" / STUDY_TABLE.format(key=key))) == 3
