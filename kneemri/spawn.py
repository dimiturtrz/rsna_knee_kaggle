"""Running a child process on a box that is also training.

Every check in this repo shells out — `arm_check` sources each arm through bash, `kit_check` and
`queue_check` ask git and the process table, `live_dataset` calls the Kaggle CLI. All of them run from a
pre-commit or pre-push hook, which is to say they run while an arm holds 27 GB of the card and saturates
the machine, which is when Windows starts refusing to *create* processes at all.
"""

import logging
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

STATUS_DLL_INIT_FAILED = 0xC0000142
"""Windows could not initialise a new process's DLLs — it never ran, so it has no result to report.

MEASURED 2026-08-21: a `git init -q` inside a unit test returned 3221225794 while `band140` held the box,
and nine tests failed in one pre-push run — two in `arm_check`, seven in `kit_check`. Every one of the
nine spawns a child; none of the 946 that passed do. The child's own exit codes are single digits, so
this value cannot collide with a real failure from anything we run.

It is a machine-load symptom wearing a result's clothes, and that is the whole reason it is handled here
rather than in the tests that surfaced it: a check that reports "the kit disagrees with the tree" because
Windows declined to start git is worse than one that reports nothing.
"""

SPAWN_RETRIES = 3
SPAWN_BACKOFF_S = 2.0
POSIX_SIGKILL = 9
"""`signal.SIGKILL`, spelled out because the name does not exist on Windows, where `killed` never reaches it."""

SPAWN_STDERR_TAIL = 2000
"""`CalledProcessError.__str__` prints the argv and the code and never the child's own stderr, so a
transient API failure reaches the reader as a bare traceback with the diagnosis inside an attribute
nobody opens. The tail is logged before the raise."""


@dataclass(frozen=True)
class Spawn:
    """`subprocess.run`, except that a child which never STARTED is retried rather than reported."""

    retries: int = SPAWN_RETRIES
    backoff_s: float = SPAWN_BACKOFF_S

    def ran(self, command: Sequence[str], **options: object) -> subprocess.CompletedProcess[str]:
        """Run `command`, retrying only the failure that means it was never created.

        ONLY `STATUS_DLL_INIT_FAILED` is retried, and deliberately so. A blanket retry would paper over
        the failures these checks exist to find — a kit that disagrees with the tree, an arm whose flags
        do not parse — by running them again until the box got quiet enough to agree. This one value
        means the process was never created, so there is no result to hide.
        """
        for attempt in range(self.retries):
            done = self.attempted(command, **options)
            if done.returncode != STATUS_DLL_INIT_FAILED:
                return self.checked(done, command, **options)
            if attempt + 1 < self.retries:
                logger.warning(
                    "windows declined to start %s (0x%X); retrying in %.0fs",
                    command[0],
                    STATUS_DLL_INIT_FAILED,
                    self.backoff_s,
                )
                time.sleep(self.backoff_s)
        message = (
            f"{command[0]} could not be started {self.retries} times "
            f"(0x{STATUS_DLL_INIT_FAILED:X}); the machine is too loaded to spawn a process"
        )
        raise OSError(message)

    def attempted(self, command: Sequence[str], **options: object) -> subprocess.CompletedProcess[str]:
        """One try, with `check` withheld — a raise here would hide the returncode this dispatches on."""
        return subprocess.run(command, **{**options, "check": False})  # type: ignore[call-overload]  # noqa: S603 - callers pass a resolved executable and a fixed argv; no shell anywhere

    def bounded(
        self, command: Sequence[str], timeout_s: float, env: Mapping[str, str] | None = None
    ) -> subprocess.CompletedProcess[str] | None:
        """Run `command` for at most `timeout_s`; on overrun kill its WHOLE tree and answer None.

        `subprocess.run(timeout=...)` kills the direct child only. An arm read through bash that launched
        a GPU job leaves that grandchild computing, orphaned, beside a live training arm — six of them on
        2026-09-08 from three `arm_check --only` calls (knee-u2aet). The tree goes, not just its root.
        """
        for _attempt in range(self.retries):
            child = subprocess.Popen(  # noqa: S603 - callers pass a resolved executable and a fixed argv; no shell anywhere
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True, env=env
            )
            try:
                stdout, stderr = child.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self.killed(child.pid)
                child.communicate()
                return None
            if child.returncode != STATUS_DLL_INIT_FAILED:
                return subprocess.CompletedProcess(list(command), child.returncode, stdout, stderr)
            time.sleep(self.backoff_s)
        message = f"{command[0]} could not be started {self.retries} times; the machine is too loaded to spawn"
        raise OSError(message)

    @staticmethod
    def killed(pid: int) -> None:
        """Kill a process and every descendant it started."""
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False)  # noqa: S603, S607 - fixed argv, a Windows system binary
            return
        os.kill(-pid, POSIX_SIGKILL)

    def checked(
        self, done: subprocess.CompletedProcess[str], command: Sequence[str], **options: object
    ) -> subprocess.CompletedProcess[str]:
        """Honour the caller's `check`, now that the result is the child's own and not Windows'."""
        if options.get("check") and done.returncode != 0:
            logger.error(
                "%s exited %d; its own stderr: %s",
                command[0],
                done.returncode,
                (done.stderr or "").strip()[-SPAWN_STDERR_TAIL:] or "<empty>",
            )
            raise subprocess.CalledProcessError(done.returncode, list(command), done.stdout, done.stderr)
        return done
