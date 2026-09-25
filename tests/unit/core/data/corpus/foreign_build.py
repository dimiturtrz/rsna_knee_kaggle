"""The shared foreign decode, on arrays and tables the test builds itself."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from core.data.corpus.foreign_build import ForeignBuild, SlotVolume
from core.data.corpus.slots import FASTMRI_SCHEME, Routing
from core.data.corpus.study_cache import StudyCache

DEPTH, MATRIX = 6, 32
SPACING = 140.0 / 320


def _build(tmp_path: Path) -> ForeignBuild:
    return ForeignBuild(StudyCache(tmp_path, slices=4, image_size=8, routing=Routing(scheme=FASTMRI_SCHEME)))


def _volume(scale: float = 1.0) -> np.ndarray:
    return np.arange(DEPTH * MATRIX * MATRIX, dtype=np.float32).reshape(DEPTH, MATRIX, MATRIX) * scale


def test_stacked(tmp_path: Path):
    """A volume becomes the cache's own shape, band-sampled and percentile-scaled by the shared reader."""
    stacked = _build(tmp_path).stacked(_volume(1e-8), SPACING)

    assert stacked.shape == (4, 8, 8), "slices asked for, at the cache's in-plane size"
    assert stacked.dtype == torch.uint8, "the cache holds eight bits per pixel and so must this"
    assert int(stacked.max()) - int(stacked.min()) > 250, (
        "scaled to its own 1st-99th percentiles — these magnitudes peak near 1e-4 and a fixed window "
        "would return an empty image"
    )


def test_stacked_crops_when_the_physical_extent_fits(tmp_path: Path):
    """The physical crop is taken when the spacing admits one and skipped when it would exceed the frame.

    Both branches matter: a wrong spacing that silently declined to crop would hand the encoder a
    different mm/pixel than every member on disk was fitted at.
    """
    build = _build(tmp_path)

    fine = build.stacked(_volume(), SPACING)
    coarse = build.stacked(_volume(), 130.0 / 20.0)

    assert coarse.shape == fine.shape == (4, 8, 8)
    assert not np.array_equal(coarse.numpy(), fine.numpy()), (
        "a 130 mm crop at 6.5 mm/px takes 20 of 32 pixels; at 0.4375 mm/px it wants 297 and cannot"
    )


def test_write(tmp_path: Path):
    """Images, index and mask land; a slot is marked present exactly where a volume was written."""
    build = _build(tmp_path)

    build.write(["a", "b"], [[SlotVolume(0, _volume(), SPACING), SlotVolume(1, _volume(), SPACING)], []])

    images, identifiers = build.cache.load()
    assert identifiers == ["a", "b"]
    assert json.loads(build.cache.index_path.read_text(encoding="utf-8")) == identifiers
    assert images.images.shape == (2, 2, 4, 8, 8)
    assert images.mask.tolist() == [[1.0, 1.0], [0.0, 0.0]], "a study with no volume is all absent"
    assert not images.images[1].any(), "and its pixels stay black"


def test_write_table(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """The table lands as written, and only the named columns it actually carries are counted."""
    destination = tmp_path / "labels" / "table.csv"
    table = pd.DataFrame({"study": ["a", "b"], "ACL": [0.08, 0.94]})

    with caplog.at_level(logging.INFO):
        ForeignBuild.write_table(table, destination, ["ACL", "Synovitis"])

    pd.testing.assert_frame_equal(pd.read_csv(destination), table)
    assert "ACL: {0.08: 1, 0.94: 1}" in caplog.text
    assert "Synovitis" not in caplog.text, "a column the table lacks is not reported"
