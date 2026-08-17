"""What the slice band discards, on stacks whose through-plane profile is known by construction."""

from pathlib import Path

import numpy as np

from kneemri.corpus.band_coverage import BandCoverage, BandRequest
from kneemri.corpus.slot_reader import Band


def _ended_stack() -> np.ndarray:
    """Ten slices, each holding tissue, so every slice contributes the same mass.

    Uniform on purpose: with equal mass per slice the discarded share is decided by the band alone, so
    the test measures the geometry rather than an intensity accident.
    """
    stack = np.zeros((10, 20, 20), np.float32)
    stack[:, 5:15, 5:15] = 1.0
    return stack


def _one_ended_stack() -> np.ndarray:
    """Tissue only in the two deepest slices — everything the 0.20-0.80 band throws away at ONE end."""
    stack = np.zeros((10, 20, 20), np.float32)
    stack[8:, 5:15, 5:15] = 1.0
    return stack


def test_profile():
    """Mass per slice, Otsu-thresholded, so an empty slice weighs nothing and a full one weighs its block."""
    mass = BandCoverage().profile(_one_ended_stack())

    assert mass.shape == (10,)
    assert mass[:8].sum() == 0.0, "empty slices carry no tissue"
    assert (mass[8:] == 100.0).all(), "a 10x10 block is a hundred pixels of tissue"


def test_discarded():
    """The share outside the band, split by end — and nothing at all from a series holding no tissue."""
    uniform = BandCoverage().discarded(_ended_stack())

    # Ten slices at linspace(0,1,10): two sit below 0.20 and two above 0.80, so a fifth is discarded.
    assert uniform["discarded"] == 0.4
    assert uniform["low_end"] == uniform["high_end"] == 0.2, "a uniform series loses each end equally"

    lopsided = BandCoverage().discarded(_one_ended_stack())
    assert lopsided["low_end"] == 0.0
    assert lopsided["high_end"] == 1.0, "tissue only in the deep tail is tissue the band deletes entirely"

    assert BandCoverage(Band(low=0.0, high=1.0)).discarded(_ended_stack())["discarded"] == 0.0
    assert BandCoverage().discarded(np.zeros((4, 20, 20), np.float32)) == {}, "no tissue, no evidence"


def test_interval():
    """A bootstrap interval on the mean, which brackets a constant sample at the constant."""
    low, high = BandCoverage().interval(np.full(50, 0.25))

    assert low == high == 0.25, "a constant sample cannot resample to anything else"

    spread_low, spread_high = BandCoverage().interval(np.linspace(0.0, 1.0, 200))
    assert spread_low < 0.5 < spread_high


def test_report():
    """The denominators lead, and each slot is one row — a share over what decoded is not a corpus share."""
    measured = [("SAG_FLUID_FS", {"discarded": 0.3, "low_end": 0.1, "high_end": 0.2, "span_mm": 100.0})] * 8
    lines = BandCoverage().report(measured, sampled=5, corpus=4407)

    assert lines[0] == "band 0.2-0.8 over 8 series from 5 studies sampled of 4407 in the corpus"
    assert sum(line.startswith("SAG_FLUID_FS") for line in lines) == 1
    row = next(line for line in lines if line.startswith("SAG_FLUID_FS"))
    assert "0.3000" in row
    # A 100 mm span cut at 0.20 loses 20 mm from each end — the share restated as the distance the
    # peripheral meniscus either does or does not sit inside.
    assert "20.0" in row


def test_measure_study(tmp_path: Path):
    """A study that will not decode is silent about the band rather than counted as discarding none."""
    request = BandRequest(identifier="absent", source=str(tmp_path), planes={})

    assert BandCoverage.measure_study(request) == []
