"""Each platform's answer about its idle timer, and which answer this machine gets."""

import sys

from core.sleep_policy import Sleeps, StaysAwake, ThisMachine


def test_sleeps_hold():
    """`Sleeps.hold` answers False rather than pretending: a run that believes it is protected and is
    not is worse off than one that knows it is unattended under the machine's own power scheme."""
    assert Sleeps().hold() is False


def test_sleeps_release():
    """Nothing was taken, so giving it back raises nothing."""
    assert Sleeps().release() is None


def test_stays_awake_hold():
    """The Windows request lands — `SetThreadExecutionState` answers non-zero, which is the only
    success signal the call has. Released in the same breath so the thread is left as it was found."""
    policy = StaysAwake()
    try:
        assert policy.hold() is True
    finally:
        policy.release()


def test_stays_awake_release():
    """Safe to release from a state it did not set, which is what `Wakelock.__exit__` does after a
    raise — and what a second release from a nested block would do."""
    assert StaysAwake().release() is None


def test_policy():
    """The dispatch is what selects a policy, so the selection is worth pinning: on win32 it must be
    the real one, and an unknown platform must get the no-op rather than a KeyError."""
    assert isinstance(ThisMachine().policy(), StaysAwake if sys.platform == "win32" else Sleeps)
    assert isinstance(ThisMachine("a platform that does not exist").policy(), Sleeps)
