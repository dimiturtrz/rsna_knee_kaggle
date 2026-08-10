from pathlib import Path

import numpy as np
import pandas as pd

from kneemri.slots import SLOTS
from kneemri.study_cache import StudyCache

from .conftest import write_series


def test_shape(tmp_path: Path):
    """The cache is [studies, slots, slices, size, size] — the slot axis is the head's axis."""
    assert StudyCache(tmp_path, slices=4, image_size=16).shape(7) == (7, len(SLOTS), 4, 16, 16)


def test_images_path(tmp_path: Path):
    """The file name states everything that decides the contents, so two configs cannot collide."""
    path = StudyCache(tmp_path, slices=4, image_size=16).images_path
    assert path.name == "images_16px_4sl.npy"


def test_mask_path(tmp_path: Path):
    assert StudyCache(tmp_path, slices=4, image_size=16).mask_path.name == "mask_16px_4sl.npy"


def test_index_path(tmp_path: Path):
    assert StudyCache(tmp_path, slices=4, image_size=16).index_path.name == "index_16px_4sl.json"


def test_planes_of(tmp_path: Path):
    """The plane is the one thing the slot predicate takes from the shipped series table."""
    table = tmp_path / "series.csv"
    pd.DataFrame({"SeriesInstanceUID": ["s1", "s2"], "Anatomical_Plane": ["Sagittal", "Axial"]}).to_csv(
        table, index=False
    )
    assert StudyCache.planes_of(table) == {"s1": "Sagittal", "s2": "Axial"}


def _built(tmp_path: Path):
    """A one-study cache, built and loaded — the fixture the corpus accessors are read from."""
    source = tmp_path / "src"
    write_series(source / "study0" / "sag", slices=4, description="SAG PD FS", size=32)
    cache = StudyCache(tmp_path / "cache", slices=2, image_size=8)
    cache.build(source, ["study0"], {"sag": "Sagittal"}, workers=1)
    corpus, _ = cache.load()
    return corpus


def test_build(tmp_path: Path):
    """Decoding writes each study into its own slice of the map, and records what was filled."""
    source = tmp_path / "src"
    write_series(source / "study0" / "sag", slices=4, description="SAG PD FS", size=32)
    cache = StudyCache(tmp_path / "cache", slices=2, image_size=8)
    cache.build(source, ["study0"], {"sag": "Sagittal"}, workers=1)
    corpus, identifiers = cache.load()
    assert corpus.images.shape == (1, len(SLOTS), 2, 8, 8)
    assert identifiers == ["study0"]
    assert corpus.mask[0].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert int(np.asarray(corpus.images[0, 0]).max()) > 0, "the filled slot carries pixels"


def test_load(tmp_path: Path):
    """Loading maps the cache read-only rather than reading 35 GB into memory."""
    source = tmp_path / "src"
    write_series(source / "study0" / "sag", slices=4, size=32)
    cache = StudyCache(tmp_path / "cache", slices=2, image_size=8)
    cache.build(source, ["study0"], {"sag": "Sagittal"}, workers=1)
    corpus, _ = cache.load()
    assert isinstance(corpus.images, np.memmap)


def test_image_size(tmp_path: Path):
    """The corpus reports its own resolution, so no consumer has to assume the default."""
    corpus = _built(tmp_path)
    assert corpus.image_size == 8


def test_slices(tmp_path: Path):
    """And its slice depth, which bounds the windows any member can be read over."""
    assert _built(tmp_path).slices == 2


def test_decode_one(tmp_path: Path):
    """One study decoded into its own row of the map, returning the mask of what was filled."""
    source = tmp_path / "src"
    write_series(source / "study0" / "sag", slices=4, description="SAG PD FS", size=32)
    cache = StudyCache(tmp_path / "cache", slices=2, image_size=8)
    cache.directory.mkdir(parents=True, exist_ok=True)
    np.lib.format.open_memmap(cache.images_path, mode="w+", dtype=np.uint8, shape=cache.shape(1))
    position, flags = StudyCache.decode_one(
        (0, "study0", str(source), str(cache.images_path), 2, 8, {"sag": "Sagittal"})
    )
    assert position == 0
    assert flags == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
