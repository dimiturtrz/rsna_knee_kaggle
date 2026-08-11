import logging

import pytest

from kneemri.heartbeat import Heartbeat


def test_tick(caplog: pytest.LogCaptureFixture):
    """A heartbeat reports on the SECOND that the interval elapses, not on a step count.

    The clock is injected so this pins the pacing without sleeping — a unit test that slept would be
    testing the scheduler and would trip the no-sleep gate.
    """
    now = [100.0]
    beat = Heartbeat(every_seconds=30.0, clock=lambda: now[0])

    with caplog.at_level(logging.INFO):
        beat.tick(1, 100, 0.5, images=864)
        now[0] += 10.0
        beat.tick(2, 100, 0.5, images=864)
        assert not caplog.records, "quiet until the interval elapses"

        now[0] += 25.0
        beat.tick(3, 100, 0.4, images=864)

    assert "step 3/100" in caplog.text
    assert "img/s" in caplog.text, "the rate is the number that says whether the card is working"
