"""Is every queued arm still gated by a live process?

`python -m kneemri.experiments.queue_check`

MEASURED 2026-08-20: `aug_long` wrote its completion marker and the card then sat at 5% for about five
minutes, because the `after.sh` watchers that were supposed to start the next three arms no longer
existed. They had been launched from a session that had since ended; `nohup` gives them PPID 1 but does
not survive whatever ends the harness shell here.

THE TRAP IS THAT THE QUEUE LOOKS ARMED. A dead watcher's log still reads `waiting for 'ARM DONE ...'`,
forever, because that line was printed before it died — so the one artefact a queue leaves behind says
the same thing whether it is waiting or gone. Nothing distinguishes the two except the process table.

So this check crosses the two: an arm is STRANDED when its watcher's log says it has not started yet
AND no live `after.sh` carries that marker in its argv. It is the queue's counterpart to `ledger_check`
— that one catches an arm that ran and was never written down, this one catches an arm that was written
down and will never run.

The process listing is passed in rather than read inside, so the crossing itself is a pure function of
two strings and the unit test needs no process to kill.
"""

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from kneemri.spawn import Spawn

logger = logging.getLogger(__name__)

WATCHER_GLOB = "after_*.log"
ARM_SUFFIX = ".sh"
HEADER = "before running: "
STARTED = "=== "


@dataclass(frozen=True)
class Probe:
    """One way to ask this machine for its process table, as a resolved executable and a fixed argv."""

    executable: str
    argv: tuple[str, ...]

    def read(self) -> str:
        """This probe's output, or empty when the executable is absent or the call fails.

        The decode is lossy on purpose: a process table here carries a shell snapshot's bytes in
        whatever the console encoding is, and one undecodable byte in an unrelated process must not
        cost the answer.
        """
        found = shutil.which(self.executable)
        if not found:
            return ""
        # resolved executable, fixed argv, no shell
        completed = Spawn().ran(
            [found, *self.argv], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
        )
        return completed.stdout


PROBES = (
    Probe("ps", ("-ef",)),
    Probe(
        "powershell",
        ("-NoProfile", "-Command", "Get-CimInstance Win32_Process | ForEach-Object { $_.CommandLine }"),
    ),
)
"""Ordered, first non-empty wins — and the second one is why this is a list rather than a call.

MEASURED 2026-08-20: `ps -ef` lists the running arm correctly in the harness shell, and
`shutil.which("ps")` from the interpreter this check runs in returns None. The MSYS `ps` lives in
git-bash's `/usr/bin`, which is not on the PATH a Windows Python process inherits, so the check
degraded to "liveness cannot be crossed" and returned success — the exact silent pass it exists to
prevent, on the only machine it runs on.

`Get-CimInstance Win32_Process` is the Windows counterpart and is strictly better here: it carries the
FULL command line, which is what the crossing matches on, where `tasklist` carries only an image name
and would match no watcher at all.
"""


@dataclass(frozen=True)
class Watch:
    """One watcher log, read for the marker it waits on and the arms it has yet to start."""

    log: str
    marker: str
    arms: tuple[str, ...]
    started: int
    """How many arms this watcher has announced — a chain runs them in order, so this is a position."""

    @classmethod
    def of(cls, path: Path) -> "Watch | None":
        """One watcher log as a watch, or None when the file is not one.

        `logs/` holds gate logs, commit logs and push logs beside these; a file whose first line is not
        the header `after.sh` prints is not a queue and must not be reported as a broken one.
        """
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines or HEADER not in lines[0]:
            return None
        marker = lines[0].split("'")[1] if "'" in lines[0] else ""
        entries = lines[0].split(HEADER, 1)[1].split()
        return cls(
            log=path.name,
            marker=marker,
            arms=tuple(entry for entry in entries if entry.endswith(ARM_SUFFIX)),
            started=sum(1 for line in lines if line.startswith(STARTED)),
        )

    def waiting(self) -> tuple[str, ...]:
        """The arms this watcher still owes, which is every one it has not announced.

        The arm it is running RIGHT NOW is not owed — it was announced — so a watcher killed mid-arm
        reports the tail it will never reach rather than the one already on the card.
        """
        return self.arms[self.started :]


class QueuedArms:
    """Every watcher log in one directory, crossed against a process listing."""

    def __init__(self, directory: Path, running: str) -> None:
        self.directory = directory
        self.running = running

    def watches(self) -> list[Watch]:
        """Each file that is a watcher log, newest first, so the live queue reads at the top."""
        found = [Watch.of(path) for path in sorted(self.directory.glob(WATCHER_GLOB), key=self._age)]
        return [watch for watch in found if watch is not None]

    def alive(self, watch: Watch) -> bool:
        """Whether some `after.sh` in the listing is still gating on this watch's marker."""
        return watch.marker in self.running

    def stranded(self) -> list[str]:
        """Arms owed by a dead watcher and owed by no live one — the queue's actual hole.

        Reported as ARMS rather than as watches because recovery relaunches a chain over the arms, and
        because the incident's own recovery left three dead logs claiming arms the new chain had taken
        over. Reporting per watch would have called that healthy queue stranded three times.

        An arm ANYONE has announced is covered too, and that second rule earned itself on this check's
        first live run: `aprior_dense` was on the card, announced by the live chain, and a dead log
        still claiming it made the arm read as stranded while it was training.
        """
        covered = {arm for watch in self.watches() if self.alive(watch) for arm in watch.waiting()}
        covered |= {arm for watch in self.watches() for arm in watch.arms[: watch.started]}
        owed = [arm for watch in self.watches() if not self.alive(watch) for arm in watch.waiting()]
        return [arm for arm in dict.fromkeys(owed) if arm not in covered]

    def lines(self) -> list[str]:
        """The queue, then the verdict — and the verdict is the last line so a tail sees it."""
        watches = self.watches()
        if not watches:
            return [f"no watcher logs under {self.directory}", "OK: nothing is queued"]
        stranded = self.stranded()
        rows = [f"{len(watches)} watcher logs under {self.directory}"]
        rows += [
            f"  {watch.log}  waits on '{watch.marker}'  {watch.started}/{len(watch.arms)} started"
            f"  {'live' if self.alive(watch) else 'no watcher'}"
            for watch in watches
        ]
        verdict = (
            f"STRANDED: {' '.join(stranded)} — owed by a dead watcher and by no live one, so relaunch "
            "after.sh over them; the marker they wait on may already be written"
            if stranded
            else "OK: every arm still owed is gated by a live process"
        )
        return [*rows, verdict]

    @staticmethod
    def listing(probes: tuple[Probe, ...] = PROBES) -> str:
        """This machine's process table as one string, or empty when no probe could read it.

        An unreadable table must not read as "nothing is running" — that turns every queued arm into a
        false strand — so the caller is told, and the check reports what it could not do instead.
        """
        for probe in probes:
            found = probe.read()
            if found:
                return found
        return ""

    @staticmethod
    def _age(path: Path) -> float:
        return -path.stat().st_mtime


def main() -> None:
    """Report any queued arm whose watcher is gone.

    EXITS NON-ZERO WHEN THE TABLE CANNOT BE READ, not only when an arm is stranded. A warning that
    leaves the status at 0 reads, in a log tail, exactly like a pass — which is the shape that has now
    cost this repo three times (bare `nox` exiting 127, a commit whose Failed line sat above the tail,
    and on 2026-08-20 a process table that could not be spawned at all). "The check could not run" and
    "the check ran and found nothing" must not share an exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("logs"), help="where the watcher logs live")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    running = QueuedArms.listing()
    if not running:
        logger.warning(
            "no process listing: none of %s could be read, so liveness cannot be crossed",
            ", ".join(probe.executable for probe in PROBES),
        )
        raise SystemExit(1)
    for line in QueuedArms(arguments.directory, running).lines():
        logger.info(line)


if __name__ == "__main__":
    main()
