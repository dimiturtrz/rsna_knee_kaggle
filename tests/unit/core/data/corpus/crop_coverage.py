"""Retained tissue under each crop centre, on stacks whose right answer is known by construction."""

from pathlib import Path

import numpy as np

from core.data.corpus.crop_coverage import CoverageRequest, CropCoverage
from core.data.corpus.study_cache import CorpusSample

EXTENTS = (130.0, 80.0)
SPACING = 1.0


def _offset_stack() -> np.ndarray:
    """A 200 mm frame at 1 mm/pixel whose tissue is a 60 mm block 30 mm off the image centre.

    Off by enough that an 80 mm box about the middle clips it and the 130 mm box does not — which is
    exactly the regime knee-1nn's tight crop lives in, at this corpus's measured p90 offset.
    """
    stack = np.zeros((3, 200, 200), np.float32)
    stack[:, 100:160, 100:160] = 1.0
    return stack


def test_tissue():
    """Otsu splits the block from the background, so the mask counts tissue and not the frame."""
    mask = CropCoverage().tissue(_offset_stack())

    assert mask.shape == (3, 200, 200)
    assert mask[:, 100:160, 100:160].all(), "the block is tissue"
    assert not mask[:, :100, :100].any(), "the background is not"


def test_series():
    """The tissue centre keeps more at the TIGHT extent, and both keep everything at the wide one."""
    kept = CropCoverage(EXTENTS).series(_offset_stack(), SPACING)

    assert set(kept) == {"image@130", "tissue@130", "image@80", "tissue@80"}
    assert kept["image@130"] == 1.0, "130 mm about the middle still contains a block 40 px off centre"
    assert kept["tissue@80"] > kept["image@80"], "at 80 mm the middle clips what the tissue centre holds"
    assert kept["tissue@80"] == 1.0

    assert CropCoverage(EXTENTS).series(np.zeros((2, 200, 200), np.float32), SPACING) == {}, "no tissue, no evidence"
    assert CropCoverage(EXTENTS).series(_offset_stack(), None) == {}, "unknown spacing crops nothing to score"


def test_paired():
    """The delta is paired over series, and starred only where its interval excludes zero."""
    coverage = CropCoverage((80.0,))
    lines = coverage.paired([{"tissue@80": 0.9 + index / 100, "image@80": 0.6 + index / 100} for index in range(20)])

    assert len(lines) == 1
    assert "+0.3000" in lines[0]
    assert lines[0].endswith("over 20 series")
    assert "]*" in lines[0], "a constant +0.3 cannot straddle zero"

    assert coverage.paired([{"image@80": 0.5}]) == [], "a series measured under one centre is not a pair"


def test_report():
    """The denominators lead, because a share over what decoded is not a share over the corpus."""
    lines = CropCoverage((80.0,)).report([{"tissue@80": 0.9, "image@80": 0.6}], CorpusSample(["a", "b"], 4407))

    assert lines[0] == "1 series measured, from 2 studies sampled of 4407 in the corpus"
    assert any(line.startswith("tissue@80") for line in lines)
    assert any(line.startswith("tissue - image at 80 mm") for line in lines)


def test_measure_study(tmp_path: Path):
    """A study that will not decode is silent about centring rather than counted as retaining none."""
    request = CoverageRequest(identifier="absent", source=str(tmp_path), planes={})

    assert CropCoverage.measure_study(request) == []
