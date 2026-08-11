from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_key(tmp_path: Path):
    """The canonical scheme keeps its historic name; any other slot count is named explicitly.

    The slot count decides the array's SHAPE, so leaving it out of the filename meant a nine-slot
    build would overwrite the six-slot cache in place. The canonical name is preserved so arrays
    already on disk stay loadable — a rename cannot be done while a trainer holds one mapped.
    """
    canonical = StudyCache(tmp_path, slices=12, image_size=336)
    assert canonical.key == "336px_12sl"
    assert canonical.images_path.name == "images_336px_12sl.npy"

    extended = StudyCache(tmp_path, slices=12, image_size=336, slots=9)
    assert extended.key == "336px_12sl_9slot"
    assert extended.images_path != canonical.images_path, "a different slot count must not collide"
    assert extended.mask_path != canonical.mask_path
    assert extended.index_path != canonical.index_path


def test_shape_follows_the_slot_count(tmp_path: Path):
    """The shape reports the cache's OWN slot count, not the module-level scheme's."""
    assert StudyCache(tmp_path, slices=12, image_size=336, slots=9).shape(4407)[1] == 9


def test_build_refuses_a_slot_count_the_decoder_cannot_fill(tmp_path: Path):
    """A nine-slot cache built by a six-slot decoder would hold three blank sequences and look fine.

    Verified by asking for it: the shape would be right, the file would be the right size, and three
    of every study's sequences would be zeros. That is the failure this project keeps meeting — the
    run completes and the result is quietly meaningless — so it is an error, not a warning.
    """
    with pytest.raises(ValueError, match="would be written as zeros"):
        StudyCache(tmp_path, slices=12, image_size=336, slots=9).build(tmp_path, ["a"], {})
