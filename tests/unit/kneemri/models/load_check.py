"""The verdict over a registry-wide load, over `TrunkLoad` values rather than over fetched weights."""

import logging
from types import SimpleNamespace

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


class _Encoder:
    """A registry entry whose `load` answers without fetching anything."""

    def __init__(self, loaded: object) -> None:
        self.dim = DIM
        self.loaded = loaded

    def load(self) -> object:
        if isinstance(self.loaded, Exception):
            raise self.loaded
        return self.loaded


def test_loaded_reads_the_width_off_the_built_trunk():
    trunk = SimpleNamespace(config=SimpleNamespace(hidden_size=384))
    load = LoadCensus.loaded("mislabelled", _Encoder(trunk))
    assert load == TrunkLoad("mislabelled", DIM, width=384)
    assert "declares dim 768" in load.defect


@pytest.mark.parametrize("refusal", [ValueError("drifted"), RuntimeError("size mismatch"), OSError("gated repo")])
def test_loaded_turns_a_refusal_into_a_defect_rather_than_a_crash(refusal: Exception):
    load = LoadCensus.loaded("radiodino", _Encoder(refusal))  # pyrefly: ignore[bad-argument-type]
    assert load.width is None
    assert load.defect == f"radiodino: {refusal}"


def test_loaded_reads_a_trunk_with_no_width_as_a_disagreement():
    load = LoadCensus.loaded("bare", _Encoder(SimpleNamespace(config=object())))
    assert load.width is None
    assert "the loaded trunk is None wide" in load.defect
