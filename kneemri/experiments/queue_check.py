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

MEASURED 2026-08-21, the opposite hole: an arm that will run TWICE. Two `after.sh` processes sat on one
marker, both owing `band_dense_llrd.sh`, and this check called the queue healthy — liveness was decided
by the marker alone, so a chain and the watcher it had superseded vouched for each other. Both failures
are silent in the same way, and both are read off the same crossing, so both live here.

The process listing is passed in rather than read inside, so the crossing itself is a pure function of
two strings and the unit test needs no process to kill.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from core import cli
from kneemri.process_table import PROBES, ProcessTable

logger = logging.getLogger(__name__)

WATCHER_GLOB = "after_*.log"
WATCHER = "after.sh"
ARM_SUFFIX = ".sh"
HEADER = "before running: "
STARTED = "=== "

DONE = "ARM DONE "
"""What `lib.sh` echoes on SUCCESS ONLY, once the trainer exited 0 and wrote a `best holdout` line.

READ FROM THE ARM'S OWN LOG, `<stem>.log`, never from a watcher's — a watcher's header quotes this
same string as the marker it waits on, so a queue that has not started anything would otherwise read
as a queue that finished. `ARM FAILED` deliberately does NOT close a strand: an arm that died still
owes its work, and relaunching it is the right advice.
"""

UNREADABLE = 1
FOUND = 2
"""Distinct exit codes, because "the table could not be read" is not "the queue is broken"."""


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

    def watchers(self) -> list[str]:
        """The lines of the process listing that are `after.sh` invocations, argv and all.

        Narrowed to the watchers on purpose: an arm that is RUNNING has a process line of its own
        carrying the same arm path, and counting that as a second gate would report every arm on the
        card as doubly queued.

        THE LAUNCHING SHELL IS DROPPED BY PARENTAGE, and it has to be: a watcher is started as
        `nohup bash experiments/after.sh …` from a harness shell whose own `-c` string embeds that whole
        command, so the table holds two lines carrying the same argv for ONE queue. Dropping any line
        that is the PARENT of another watcher line removes exactly that wrapper, while two genuinely
        separate watchers — neither one's parent — both survive to be counted.
        """
        lines = [line for line in self.running.splitlines() if WATCHER in line]
        parents = {ProcessTable.parentage(line)[1] for line in lines} - {""}
        return [line for line in lines if ProcessTable.parentage(line)[0] not in parents]

    def alive(self, watch: Watch) -> bool:
        """Whether a live `after.sh` carries this watch's marker AND the exact arms it names.

        MATCHED AS THE LINE'S TAIL, not as a substring, and that is the whole correction. `after.sh`
        takes its arms last, so a one-arm watcher's argv is a SUFFIX of the two-arm chain that replaced
        it — and on 2026-08-21 a substring test let the relaunched chain vouch for the log of a watcher
        that had already been killed. A dead queue that reads live is the exact failure this check was
        written for.
        """
        return any(watch.marker in line and self._armed(line) == watch.arms for line in self.watchers())

    def doubled(self) -> list[str]:
        """Arms that more than one live watcher will start — the expensive shape, so it is BLOCKING.

        MEASURED 2026-08-21: two `after.sh` processes sat on the same marker, both owing
        `band_dense_llrd.sh`, and this check called the queue healthy. Had the marker fired, one arm
        would have launched twice on one card — 2.5 h of slot, and two arms of that geometry together
        drove the card to 31.5 GB of its 32 (the measurement `after.sh`'s own header cites).

        Counted over the PROCESS TABLE rather than over the logs, because that is what will actually
        run: two watchers whose chains merely overlap are as costly as two identical ones, and a log
        says nothing about whether its writer is still there.
        """
        owed = dict.fromkeys(arm for watch in self.watches() for arm in watch.waiting())
        return [arm for arm in owed if sum(1 for line in self.watchers() if arm in line) > 1]

    def landed(self, arm: str) -> bool:
        """Whether this arm's own log already carries its completion marker — the work is DONE.

        THE CHECK REASONS ABOUT WATCHERS, AND AN ARM IS NOT A WATCHER. MEASURED 2026-08-24: it printed
        STRANDED for `cnx_own_recipe.sh`, which had already finished — `ARM DONE` in its log, best
        0.8186 at e6, checkpoint on disk, written up in the ledger. Obeying the advice would have put a
        finished 3.1 h arm back on the card, on top of `small_batch6_ff` at 6/12. Its own hedge named the
        wrong hazard too: it warns the MARKER may already be written, when what matters is that the ARM
        already ran.

        Two categories, two different actions, and only one of them touches the GPU.
        """
        stem = Path(arm).stem
        log = self.directory / f"{stem}.log"
        if not log.is_file():
            return False
        return f"{DONE}{stem}" in log.read_text(encoding="utf-8", errors="replace")

    def finished(self) -> list[str]:
        """Arms nobody owes any more because they already ran — a watcher log to clean up, not work.

        NOT A FAULT, on purpose. Nothing here is wrong with the card or the queue; a dead watcher whose
        arms all landed is litter. Reporting it as blocking would train the reader to ignore a check
        whose other two verdicts cost a card.
        """
        return [arm for arm in self._owed_by_the_dead() if self.landed(arm)]

    def stranded(self) -> list[str]:
        """Arms owed by a dead watcher and owed by no live one — the queue's actual hole.

        Reported as ARMS rather than as watches because recovery relaunches a chain over the arms, and
        because the incident's own recovery left three dead logs claiming arms the new chain had taken
        over. Reporting per watch would have called that healthy queue stranded three times.

        An arm ANYONE has announced is covered too, and that second rule earned itself on this check's
        first live run: `aprior_dense` was on the card, announced by the live chain, and a dead log
        still claiming it made the arm read as stranded while it was training.

        AND AN ARM WHOSE WORK ALREADY LANDED IS NOT OWED AT ALL — see `landed`, which is the third rule
        and the one that would have cost the card rather than a reading.
        """
        covered = {arm for watch in self.watches() if self.alive(watch) for arm in watch.waiting()}
        covered |= {arm for watch in self.watches() for arm in watch.arms[: watch.started]}
        owed = self._owed_by_the_dead()
        return [arm for arm in owed if arm not in covered and not self.landed(arm)]

    def _owed_by_the_dead(self) -> list[str]:
        """Every arm a watcher with no live process still names, in order and without repeats."""
        owed = [arm for watch in self.watches() if not self.alive(watch) for arm in watch.waiting()]
        return list(dict.fromkeys(owed))

    def faults(self) -> list[str]:
        """Every way this queue is wrong, or empty when it is right.

        The two failures are opposite and both are silent: an arm NO live watcher will start, and an arm
        SEVERAL of them will. One wastes a card that is idle, the other oversubscribes a card that is
        not.
        """
        found = []
        if doubled := self.doubled():
            found.append(
                f"DOUBLE-QUEUED: {' '.join(doubled)} — more than one live after.sh will start each of "
                "these, so kill all but one chain; two arms of one geometry do not fit the card together"
            )
        if stranded := self.stranded():
            found.append(
                f"STRANDED: {' '.join(stranded)} — owed by a dead watcher and by no live one, so relaunch "
                "after.sh over them; the marker they wait on may already be written"
            )
        return found

    def lines(self) -> list[str]:
        """The queue, then the verdict — and the verdict is the last line so a tail sees it."""
        watches = self.watches()
        if not watches:
            return [f"no watcher logs under {self.directory}", "OK: nothing is queued"]
        rows = [f"{len(watches)} watcher logs under {self.directory}"]
        rows += [
            f"  {watch.log}  waits on '{watch.marker}'  {watch.started}/{len(watch.arms)} started"
            f"  {'live' if self.alive(watch) else 'no watcher'}"
            for watch in watches
        ]
        if finished := self.finished():
            rows.append(
                f"  ({' '.join(finished)} already ran — a dead watcher log to clean up, not an arm to relaunch)"
            )
        return [*rows, *(self.faults() or ["OK: every arm still owed is gated by exactly one live process"])]

    @staticmethod
    def _armed(line: str) -> tuple[str, ...]:
        """The arms one `after.sh` process line carries, read the same way the log's header is read.

        Picked out by the `.sh` suffix rather than by argv position, so an arm carrying its own argument
        ("resolution.sh 448") loses that argument on BOTH sides of the comparison instead of on one.
        """
        tail = line.split(WATCHER, 1)[1]
        return tuple(token.strip("\"'") for token in tail.split() if token.endswith(ARM_SUFFIX))

    @staticmethod
    def _age(path: Path) -> float:
        return -path.stat().st_mtime


def main() -> None:
    """Report any queued arm whose watcher is gone, or whose watchers are two.

    EXITS NON-ZERO ON A FAULT AND ON AN UNREADABLE TABLE, under different codes. A warning that
    leaves the status at 0 reads, in a log tail, exactly like a pass — which is the shape that has now
    cost this repo three times (bare `nox` exiting 127, a commit whose Failed line sat above the tail,
    and on 2026-08-20 a process table that could not be spawned at all). "The check could not run" and
    "the check ran and found nothing" must not share an exit code.
    """
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--directory", type=Path, default=Path("logs"), help="where the watcher logs live")
    arguments = command.parsed()
    running = ProcessTable.read().text
    if not running:
        logger.warning(
            "no process listing: none of %s could be read, so liveness cannot be crossed",
            ", ".join(probe.executable for probe in PROBES),
        )
        raise SystemExit(UNREADABLE)
    queue = QueuedArms(arguments.directory, running)
    for line in queue.lines():
        logger.info(line)
    if queue.faults():
        raise SystemExit(FOUND)


if __name__ == "__main__":
    main()
