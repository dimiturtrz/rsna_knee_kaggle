"""Hold the machine awake for as long as a long job is running.

SCOPED TO THE JOB, NOT FIXED IN THE MACHINE. `powercfg /change standby-timeout-ac 0` would also stop
the suspends that cost three runs 20.8 hours (knee-wyzj, and see `sleep_policy` for the mechanism), and
would keep stopping them long after the run that needed it — a global setting standing in for a
property of one process. This ends when the job ends, including when the job raises, which is the case
a power-scheme edit made by hand never covers.

The request is per-THREAD (`SetThreadExecutionState`), so it has to be taken on the thread that runs
the loop and released there; a wakelock handed to a worker keeps nothing awake once that worker exits.
"""

import logging
from typing import Self

from core.sleep_policy import SleepPolicy, ThisMachine

logger = logging.getLogger(__name__)


class Wakelock:
    """A held request that the system not suspend, for the block it wraps."""

    def __init__(self, policy: SleepPolicy | None = None) -> None:
        self.policy = policy if policy is not None else ThisMachine().policy()

    def __enter__(self) -> Self:
        """Take the lock, and SAY whether it was taken — a wakelock that silently failed is worse than
        none, because the run is then unattended under a guarantee nobody has."""
        logger.info("wakelock: %s", "held" if self.policy.hold() else "unavailable on this platform")
        return self

    def __exit__(self, *_raised: object) -> None:
        """Release on the way out, including out of a raise: the machine's power behaviour must not
        depend on whether the run it was taken for succeeded."""
        self.policy.release()
