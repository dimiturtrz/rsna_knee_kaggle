"""What the slice band discards, on stacks whose through-plane profile is known by construction."""

from pathlib import Path

import numpy as np

from core.data.corpus.audit import AuditRequest
from core.data.corpus.band_coverage import BandCoverage, SlotShares
from core.data.corpus.slot_reader import Band
from core.data.corpus.study_cache import CorpusSample


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


def _shares(discarded: np.ndarray) -> SlotShares:
    ends = np.zeros_like(discarded)
    return SlotShares("SAG_T1", discarded, ends, ends, 100.0, np.full(discarded.size, 100.0))


def test_interval():
    """A bootstrap interval on the mean, which brackets a constant sample at the constant."""
    low, high = _shares(np.full(50, 0.25)).interval()

    assert low == high == 0.25, "a constant sample cannot resample to anything else"

    spread_low, spread_high = _shares(np.linspace(0.0, 1.0, 200)).interval()
    assert spread_low < 0.5 < spread_high


def test_of():
    """One slot's rows gathered once — and a span nobody measured stays zero rather than averaging in."""
    measured = [
        ("SAG_T1", {"discarded": 0.3, "low_end": 0.1, "high_end": 0.2, "span_mm": 100.0}),
        ("SAG_T1", {"discarded": 0.5, "low_end": 0.2, "high_end": 0.3, "span_mm": 0.0}),
        ("COR_T1", {"discarded": 0.9, "low_end": 0.4, "high_end": 0.5, "span_mm": 50.0}),
    ]

    shares = SlotShares.of("SAG_T1", measured)

    assert shares.discarded.tolist() == [0.3, 0.5], "the other slot's series belong to the other slot"
    assert shares.span_mm == 100.0, "a series with no spacing tag abstains rather than pulling the mean to zero"
    assert SlotShares.of("COR_T1", measured).high_end.tolist() == [0.5]


def test_row():
    """The share restated as the distance: a 100 mm span cut at 0.20 loses 20 mm from each end."""
    shares = SlotShares("SAG_T1", np.full(4, 0.3), np.full(4, 0.1), np.full(4, 0.2), 100.0, np.full(4, 100.0))

    row = shares.row(BandCoverage().band)

    assert row.startswith("SAG_T1")
    assert "0.3000" in row
    assert "20.0" in row
    assert row.endswith("     4"), "the row states how many series it averaged"


def test_spread():
    """How differently the SAME fraction reads across studies — the question a mean span cannot answer."""
    identical = SlotShares("SAG_T1", np.full(4, 0.3), np.full(4, 0.1), np.full(4, 0.2), 100.0, np.full(4, 100.0))

    assert identical.spread().split() == ["100.0", "100.0", "1.00"], "one protocol, one distance, ratio one"

    varied = SlotShares(
        "SAG_T1", np.full(101, 0.3), np.full(101, 0.1), np.full(101, 0.2), 100.0, np.linspace(50.0, 150.0, 101)
    )
    assert varied.spread().split() == ["60.0", "140.0", "2.33"], "p10 and p90 of the spans, and their ratio"

    unknown = SlotShares("SAG_T1", np.full(4, 0.3), np.full(4, 0.1), np.full(4, 0.2), 0.0, np.array([]))
    assert unknown.spread().split() == ["-", "-", "-"], "no spacing tag reports unknown, never a fabricated ratio"


def test_report():
    """The denominators lead, and each slot is one row — a share over what decoded is not a corpus share."""
    measured = [("SAG_FLUID_FS", {"discarded": 0.3, "low_end": 0.1, "high_end": 0.2, "span_mm": 100.0})] * 8
    lines = BandCoverage().report(measured, CorpusSample(["a", "b", "c", "d", "e"], 4407))

    assert lines[0] == "band 0.2-0.8: 8 series measured, from 5 studies sampled of 4407 in the corpus"
    assert sum(line.startswith("SAG_FLUID_FS") for line in lines) == 1
    row = next(line for line in lines if line.startswith("SAG_FLUID_FS"))
    assert "0.3000" in row
    # A 100 mm span cut at 0.20 loses 20 mm from each end — the share restated as the distance the
    # peripheral meniscus either does or does not sit inside.
    assert "20.0" in row


def test_measure_study(tmp_path: Path):
    """A study that will not decode is silent about the band rather than counted as discarding none."""
    request = AuditRequest(identifier="absent", source=str(tmp_path), planes={})

    assert BandCoverage().measure_study(request) == []
