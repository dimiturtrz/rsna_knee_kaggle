import pytest

from kneemri.training import card_memory as card_memory_module
from kneemri.training.card_memory import CardMemory


def test_remember():
    """The baseline is taken once and never overwritten, because only the FIRST reading is a baseline.

    Equivalence classes: unread, read once, read again after the card has moved. The third is the one
    that matters — a baseline that follows the current reading can never show anything was lost.
    """
    readings = iter([30000, 7000, 7000])
    card = CardMemory(read=lambda: next(readings))

    card.remember()
    assert card.at_start == 30000, "the first reading is the baseline"

    card.remember()
    assert card.at_start == 30000, "a later call must not overwrite it, or a loss becomes invisible"


def test_verdict():
    """The three readings a stall can produce, which have three different responses.

    Unreadable says nothing. A small difference is our own allocator and indicts the recipe. A large one
    is a process that arrived after us and indicts the scheduling — 2026-08-24, where 4.8 GB appeared
    ninety seconds into a 72-minute arm.
    """
    assert "could not be read" in CardMemory(read=lambda: None).verdict()

    ours = CardMemory(read=lambda: 5300)
    ours.remember()
    assert "no foreign process arrived" in ours.verdict(), "500 MiB of drift is our own footprint"
    assert "recipe" in ours.verdict(), "the reading is useless without the response it implies"

    foreign = CardMemory(read=iter([30000, 4800]).__next__)
    foreign.remember()
    verdict = foreign.verdict()
    assert "25200 MiB vanished" in verdict, "the LOSS is the diagnostic number, not the current level"
    assert "another process" in verdict


def test_free_mib(monkeypatch: pytest.MonkeyPatch):
    """No driver means None, not zero — and the distinction is the whole point.

    A zero would read as "the card is full", which is exactly the conclusion the caller is trying to
    establish, so a missing tool would manufacture the finding. Only the no-driver class is exercised
    here: the reading class needs a device, which a unit test may not touch.
    """
    monkeypatch.setattr(card_memory_module.shutil, "which", lambda _: None)
    assert CardMemory.free_mib() is None
