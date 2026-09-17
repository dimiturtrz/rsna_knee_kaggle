"""`StudyRender` draws a named study out of a cache it did not build."""

import json
from pathlib import Path

import numpy as np
import pytest

from core.data.corpus.study_cache import StudyCache
from kneemri.evaluation.study_render import TILE, StudyRender

SLICES = 4
SIZE = 8
STUDY = "1.2.3"


def _cache(tmp_path: Path) -> StudyCache:
    return StudyCache(tmp_path, slices=SLICES, image_size=SIZE)


def _written(tmp_path: Path, populated: list[int]) -> StudyCache:
    """A one-study cache on disk, populated on exactly the named slots."""
    cache = _cache(tmp_path)
    slots = len(cache.scheme_slots)
    images = np.full((1, slots, SLICES, SIZE, SIZE), 7, dtype=np.uint8)
    mask = np.zeros((1, slots), dtype=np.float32)
    mask[0, populated] = 1.0
    np.save(cache.images_path, images)
    np.save(cache.mask_path, mask)
    cache.index_path.write_text(json.dumps([STUDY]), encoding="utf-8")
    return cache


def test_over(tmp_path: Path) -> None:
    """It maps each recorded UID to its row."""
    render = StudyRender.over(_written(tmp_path, [0]))
    assert render.order == {STUDY: 0}


def test_page(tmp_path: Path) -> None:
    """One row per POPULATED slot, `columns` tiles across — empty slots are not drawn."""
    render = StudyRender.over(_written(tmp_path, [0, 2]))
    page = render.page(STUDY, columns=3)
    assert page.shape == (2 * TILE, 3 * TILE)


def test_page_refuses_an_empty_study(tmp_path: Path) -> None:
    """A study no slot populated cannot be drawn, and says so rather than raising on an empty concat."""
    render = StudyRender.over(_written(tmp_path, []))
    with pytest.raises(ValueError, match="no populated slot"):
        render.page(STUDY)


def test_write(tmp_path: Path) -> None:
    """It leaves the page at `<out>/<study>.png`, creating the directory."""
    render = StudyRender.over(_written(tmp_path, [1]))
    path = render.write(STUDY, tmp_path / "pages")
    assert path == tmp_path / "pages" / f"{STUDY}.png"
    assert path.is_file()
