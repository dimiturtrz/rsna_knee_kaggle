import logging

import pytest

from kneemri.training.card_memory import CardMemory
from kneemri.training.heartbeat import Heartbeat


def blind_card() -> CardMemory:
    """A card that cannot be read — the only kind a unit test may hold.

    Every test here injects one. A default `CardMemory` shells out to `nvidia-smi` the moment a stall
    latches, which is a unit test reaching for a device it did not create.
    """
    return CardMemory(read=lambda: None)


def test_tick(caplog: pytest.LogCaptureFixture):
    """A heartbeat reports on the SECOND that the interval elapses, not on a step count.

    The clock is injected so this pins the pacing without sleeping — a unit test that slept would be
    testing the scheduler and would trip the no-sleep gate.
    """
    now = [100.0]
    beat = Heartbeat(every_seconds=30.0, clock=lambda: now[0], card=blind_card())
    asked = []

    def loss(value: float):
        def read() -> float:
            asked.append(value)
            return value

        return read

    with caplog.at_level(logging.INFO):
        beat.tick(1, 100, loss(0.5), images=864)
        now[0] += 10.0
        beat.tick(2, 100, loss(0.5), images=864)
        assert not caplog.records, "quiet until the interval elapses"
        assert not asked, "a step that prints nothing must not pay for the loss — reading it synchronises"

        now[0] += 25.0
        beat.tick(3, 100, loss(0.4), images=864)

    assert asked == [0.4], "read exactly once, on the step that printed"
    assert "step 3/100" in caplog.text
    assert "loss 0.4000" in caplog.text
    assert "img/s" in caplog.text, "the rate is the number that says whether the card is working"


def test_watch(caplog: pytest.LogCaptureFixture):
    """A collapse against the run's own peak is announced once, and so is its recovery.

    The equivalence classes are the four a rate can fall into: establishing the peak, slow but not yet
    for long enough, slow for long enough, and back. The 2026-08-19 failure is the third — it ran at a
    tenth of its peak for the rest of the arm and nothing said anything.
    """
    beat = Heartbeat(card=blind_card())
    with caplog.at_level(logging.INFO):
        beat.watch(1000.0)
        beat.watch(900.0)
        assert not caplog.records, "a rate near the peak is not a stall"

        beat.watch(40.0)
        beat.watch(40.0)
        assert not caplog.records, "two slow reports could be an eval or a checkpoint write"

        beat.watch(40.0)
        assert sum(record.levelno == logging.WARNING for record in caplog.records) == 1
        assert "STALLED" in caplog.text
        assert "peak of 1000" in caplog.text, "the reference has to be in the line, or it cannot be judged"

        beat.watch(40.0)
        said = sum(record.levelno == logging.WARNING for record in caplog.records)
        assert said == 1, "said once, not per report"

        beat.watch(950.0)
        assert "recovered" in caplog.text
