"""What each platform can be asked about its idle timer, and which policy this one gets.

Windows' idle timer is driven by INPUT, not by load: a card held at 100% for eight hours resets
nothing, so an unattended run on this machine's scheme (AC "sleep after" = 7200 s, read 2026-08-18)
suspends mid-epoch and resumes hours later. Three runs lost 20.8 hours of wall-clock to exactly that
shape — one epoch of several hours surrounded by normal ones (knee-wyzj) — and it reads as a stalled
epoch rather than as a suspend, because an epoch records WALL-CLOCK and a suspended machine spends it
without computing anything.

A STRATEGY, not a platform check at the call site: the loop reads the same everywhere, and a platform
gains a real policy by joining `POLICIES` rather than by growing an `if`. `Wakelock` is the consumer
that holds one of these for the span of a block.
"""

import ctypes
import sys
from dataclasses import dataclass
from typing import Protocol

ES_CONTINUOUS = 0x80000000
"""Set the state until it is set again, rather than resetting the idle timer once."""

ES_SYSTEM_REQUIRED = 0x00000001
"""Keep the SYSTEM up. Deliberately not ES_DISPLAY_REQUIRED: a run has no reason to hold a monitor on,
and the display sleeping is not what costs an epoch."""


class SleepPolicy(Protocol):
    """What one platform can do about its idle timer, for the duration of a block."""

    def hold(self) -> bool:
        """Ask the system to stay awake. False when this platform cannot be asked."""
        ...

    def release(self) -> None:
        """Give the idle timer back, whatever `hold` managed."""
        ...


@dataclass(frozen=True)
class Sleeps:
    """The platforms whose idle timer this does not address: the job runs, the machine does as it likes."""

    def hold(self) -> bool:
        """Nothing to ask, so nothing is held."""
        return False

    def release(self) -> None:
        """Nothing was taken."""
        return


@dataclass(frozen=True)
class StaysAwake:
    """Windows: the run declares itself system-required until it is done.

    `SetThreadExecutionState` returns the PREVIOUS state and 0 on failure, which is the only error
    signal it offers — there is no exception to catch, so a failed request has to be read off the
    return value or it passes for a held lock.
    """

    def hold(self) -> bool:
        """Declare this thread's work system-required, and say whether the declaration landed."""
        return self._kernel32().SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) != 0

    def release(self) -> None:
        """Back to ES_CONTINUOUS alone — the state with no requirement in it, which is the default."""
        self._kernel32().SetThreadExecutionState(ES_CONTINUOUS)

    @staticmethod
    def _kernel32() -> ctypes.WinDLL:
        """Resolved per call rather than at import: this module is imported on every platform."""
        return ctypes.WinDLL("kernel32", use_last_error=True)


POLICIES: dict[str, type[SleepPolicy]] = {"win32": StaysAwake}
"""Idle-timer policy by `sys.platform`. A declaration-only dispatch table, the same shape as `CENTRES`
in `crop_centre.py`; anything absent from it sleeps as its owner configured."""


@dataclass(frozen=True)
class ThisMachine:
    """Which policy the running platform gets.

    The platform is a FIELD defaulting to the real one rather than read inside the lookup, so the
    branch every non-Windows machine takes is reachable from a test on this one.
    """

    platform: str = sys.platform

    def policy(self) -> SleepPolicy:
        """The declared policy, or the no-op one wherever none is declared."""
        return POLICIES.get(self.platform, Sleeps)()
