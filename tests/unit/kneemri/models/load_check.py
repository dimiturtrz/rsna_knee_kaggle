"""The verdict over a registry-wide load, over `TrunkLoad` values rather than over fetched weights."""

import logging

import pytest

from kneemri.models.load_check import LoadCensus, TrunkLoad

DIM = 768

CLEAN = TrunkLoad("dinov2base", DIM, width=DIM)
REFUSED = TrunkLoad("radiodino", DIM, width=None, failure="left 100.0% at its random initialisation")
NARROW = TrunkLoad("mislabelled", DIM, width=384)


def test_defect():
    """A clean load says nothing; a refusal and a width disagreement both name the entry."""
    assert CLEAN.defect == ""
    assert REFUSED.defect.startswith("radiodino: left 100.0%")
    assert "declares dim 768" in NARROW.defect


def test_defects():
    """Only the entries whose weights did not arrive as the entry describes them."""
    census = LoadCensus((CLEAN, REFUSED, NARROW))

    assert len(census.defects) == 2
    assert LoadCensus((CLEAN,)).defects == ()


def test_report(caplog: pytest.LogCaptureFixture):
    """Nonzero over a defective registry, zero over a clean one, and it says which entries."""
    with caplog.at_level(logging.INFO):
        assert LoadCensus((CLEAN, REFUSED)).report() == 1
        assert LoadCensus((CLEAN,)).report() == 0

    assert "radiodino" in caplog.text
    assert "every one built from its own weights" in caplog.text
