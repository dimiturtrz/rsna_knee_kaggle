"""The held request: that it is taken, that it is said out loud, and that it is always given back."""

import pytest

from core.sleep_policy import ThisMachine
from core.wakelock import Wakelock


class Records:
    """A policy that answers like a real one and remembers the order it was called in."""

    def __init__(self, *, holds: bool) -> None:
        self.holds = holds
        self.calls: list[str] = []

    def hold(self) -> bool:
        self.calls.append("hold")
        return self.holds

    def release(self) -> None:
        self.calls.append("release")


def test_the_lock_holds_on_entry_and_releases_on_exit():
    policy = Records(holds=True)
    with Wakelock(policy) as lock:
        assert lock.policy is policy
        assert policy.calls == ["hold"]
    assert policy.calls == ["hold", "release"]


def test_the_lock_is_released_even_when_the_block_raises():
    """The whole point of the context manager. A fit that dies mid-epoch must not leave the machine
    declared system-required for the rest of the session."""
    policy = Records(holds=True)
    try:
        with Wakelock(policy):
            raise RuntimeError
    except RuntimeError:
        pass
    assert policy.calls == ["hold", "release"]


def test_a_failed_hold_is_said_out_loud(caplog: pytest.LogCaptureFixture):
    """A wakelock that silently failed is the worst of the three states, because the run then goes
    unattended under a guarantee nobody is making."""
    with caplog.at_level("INFO"), Wakelock(Records(holds=False)):
        pass
    assert "unavailable" in caplog.text


def test_the_default_policy_is_the_platform_s_own():
    """Constructed with nothing, it resolves through `ThisMachine` — the lock does not carry a
    second copy of the platform decision."""
    lock = Wakelock()
    try:
        assert isinstance(lock.policy, type(ThisMachine().policy()))
    finally:
        lock.policy.release()
