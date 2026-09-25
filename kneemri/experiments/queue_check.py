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

MEASURED 2026-08-26, a third hole that both of the above read as healthy: a watcher armed on a log path
that will never exist. `blendsix_smoke.sh` was gated on `ARM DONE nocycle224_long` in
`…/nocycle224_long.full.log`, but an arm's full log carries the SEED — the real file is
`nocycle224_long_s2026.full.log`. The marker was right and only the path was wrong, and `after.sh` polls
a file, so a path that never appears is indistinguishable from a producer still working: the watcher
printed its usual `waiting for` line, one live process carried it, and this check reported OK. It would
have cost `blendsix`'s seven hours. The crossing that catches it is the same one: the watched file is
MISSING while the arm named in the marker is ON THE CARD. A producer that has not started yet is the
legitimate case and stays silent, which is exactly what separates the two.

MEASURED 2026-08-26, a fourth hole, and this one broke the GREEN verdict rather than a red one. The live
chain owing `calib_dense_hold` and `ema_dense` had written its watcher log to `experiments/logs/` while
this check globs `logs/`, so it read eleven dead logs from a finished morning, found every arm in them
already landed, and printed `OK: every arm still owed is gated by exactly one live process`. It was
looking in the wrong place and said so in the language of a pass. The crossing that catches it is the
mirror of `stranded`: a live `after.sh` in the table whose arms match no log here.

MEASURED 2026-08-26 as well: `misarmed` fired on a HEALTHY queue, because `after.sh` runs under git-bash
and records `/d/…` while Windows Python resolves that to a nonexistent `\\d\\…`. A blocking verdict whose
advice was to re-arm — that is, to tear down a live chain — over a file plainly on disk. See `_local`.

MEASURED 2026-09-11, a fifth hole, and it had swallowed the dominant case for weeks: the check could not
see a BESPOKE watcher at all. `WATCHER = "after.sh"` was matched by suffix, so `after_holdclean.sh` — and
the ~40 other `after_*.sh` scripts in `experiments/` — were not watchers to it; and `Watch.of` returns
None for any log with no `before running: ` header, which a bespoke watcher never prints. Both eyes shut
on the same queue, so it printed `no watcher logs under logs` and then `OK: nothing is queued` with TWO
live watchers owing one arm — the DOUBLE-QUEUED shape this module was written for in August, invisible
because the generic chain script is one script out of forty. `bespoke` and `duplicated` read the same
crossing off the PROCESS side alone, which is the only side a bespoke watcher writes to.

The process listing is passed in rather than read inside, so the crossing itself is a pure function of
two strings and the unit test needs no process to kill.
"""

import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from core import cli
from kneemri.process_table import PROBES, ProcessTable

logger = logging.getLogger(__name__)

WATCHER_GLOB = "after_*.log"
WATCHER = "after.sh"
WATCHER_PREFIX = "after"
"""A watcher is ANY `after*.sh`, not only the generic chain script.

MEASURED 2026-09-11, and it had been silent for weeks. `WATCHER` was matched by SUFFIX, so
`after_holdclean.sh` — and the thirty-odd other bespoke watchers in `experiments/` — were not watchers
to this check at all. Both of its eyes were shut on the same queue: `watchers()` saw no process because
no token ended in `after.sh`, and `watches()` saw no log because a bespoke watcher prints no
`before running:` header. So it printed `no watcher logs under logs` and then `OK: nothing is queued`
while TWO live watchers owed one arm — the DOUBLE-QUEUED shape this module was written for in August,
going unreported because the queue was written in the form the repo actually uses. The generic
`after.sh` is one script out of about forty.
"""
INTERPRETERS = ("bash", "bash.exe", "sh", "sh.exe")
ARM_SUFFIX = ".sh"
HEADER = "before running: "
WATCHED = "' in "
STARTED = "=== "

DONE = "ARM DONE "
PID = "pid:"
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
    watched: str
    """The file `after.sh` polls for the marker — the one thing nothing used to read.

    Split off the header's LAST `' in `, not its first: the marker is quoted and a marker containing
    that string would otherwise take the path with it.
    """
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
        head, _, tail = lines[0].partition(HEADER)
        return cls(
            log=path.name,
            marker=marker,
            watched=head.rsplit(WATCHED, 1)[1].strip() if WATCHED in head else "",
            arms=tuple(entry for entry in tail.split() if cls._is_arm(entry)),
            started=sum(1 for line in lines if line.startswith(STARTED)),
        )

    @staticmethod
    def _is_arm(entry: str) -> bool:
        """A chained `.sh` that is WORK — a chained watcher (`after_room.sh <mib> <arm>`) is a gate.

        MEASURED 2026-09-25: `after_labelcut.log` chained `after_room.sh 23000 labelcut_v3_ema.sh`, and the
        wrapper read as an owed arm, STRANDED, four days after the arm it wrapped had run under another
        watcher. The work is the arm the gate wraps, which that chain names too.
        """
        return entry.endswith(ARM_SUFFIX) and not PurePosixPath(entry).name.startswith(WATCHER_PREFIX)

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

        A LINE MUST *BE* `after.sh`, NOT MERELY NAME IT — see `_invokes`. Naming was the test until
        2026-08-30, and it reported a false DOUBLE-QUEUED on a healthy three-arm queue.

        The parentage drop below is kept as a second line of defence for the POSIX probe, where a
        launcher really is its watcher's parent. It cannot carry the rule alone: on the Windows probe
        `nohup … & disown` re-parents through an intermediate, so the launcher's pid is not the
        watcher's parent (measured: 30016 against a parent of 37388) and nothing gets dropped.
        """
        lines = [line for line in self.running.splitlines() if self._invokes(line)]
        parents = {ProcessTable.parentage(line)[1] for line in lines} - {""}
        return [line for line in lines if ProcessTable.parentage(line)[0] not in parents]

    @staticmethod
    def _watcher_at(line: str) -> int:
        """Where the watcher script sits in this process line, or -1 when the line is not a watcher.

        One place decides what a watcher IS, because three callers used to decide it separately and two
        of them used a different rule from the third.
        """
        tokens = line.split()
        if "-c" in tokens:
            return -1
        names = [PurePosixPath(token.strip("\"'").replace("\\", "/")).name for token in tokens]
        return next(
            (
                index
                for index, name in enumerate(names)
                if index > 0
                and name.startswith(WATCHER_PREFIX)
                and name.endswith(ARM_SUFFIX)
                and names[index - 1] in INTERPRETERS
            ),
            -1,
        )

    @staticmethod
    def _mentions_watcher(line: str) -> bool:
        """Whether any token here NAMES a watcher script, wherever it sits.

        Used only to exclude lines from the producer search. Broader than `_watcher_at` on purpose: a
        launcher's `-c` string is not a watcher but still carries its marker, and a producer search that
        matched it would let a watcher vouch for itself.
        """
        return any(
            PurePosixPath(token.strip("\"'").replace("\\", "/")).name.startswith(WATCHER_PREFIX)
            and token.strip("\"'").endswith(ARM_SUFFIX)
            for token in line.split()
        )

    @staticmethod
    def _invokes(line: str) -> bool:
        """Whether this process IS a watcher, rather than a process whose argv mentions one.

        MEASURED 2026-08-30 on a queue that was correct. Two kinds of impostor were being counted,
        and both are ordinary traffic rather than anything exotic:

        THE LAUNCHER. A watcher is started as `nohup bash experiments/after.sh … & disown` from a
        harness shell whose own `-c` string embeds that entire command, so the table holds two lines
        naming the same arms for ONE queue. That is what fired here: `recal_dense` was owed by exactly
        one live watcher and reported as doubly queued, which would have sent someone to kill a chain
        that had nothing wrong with it.

        THE ONLOOKER. Any grep, editor or checker whose command line carries `after.sh` matched too —
        including the very probe used to diagnose this. Those lines have no pid columns to parse, so
        `parentage` reads blank and the drop below could never remove them either.

        Both are excluded by asking WHERE the token sits. A shell running a `-c` string is a launcher
        and never a watcher, whatever it embeds; and in a real invocation the script is the token
        immediately after the interpreter, which is the one arrangement both probe formats share
        (`bash experiments/after.sh …`, quoted executable or not). An onlooker's mention is preceded
        by something else entirely.
        """
        return QueuedArms._watcher_at(line) >= 0

    def alive(self, watch: Watch) -> bool:
        """Whether a live `after.sh` carries this watch's marker AND the exact arms it names.

        MATCHED AS THE LINE'S TAIL, not as a substring, and that is the whole correction. `after.sh`
        takes its arms last, so a one-arm watcher's argv is a SUFFIX of the two-arm chain that replaced
        it — and on 2026-08-21 a substring test let the relaunched chain vouch for the log of a watcher
        that had already been killed. A dead queue that reads live is the exact failure this check was
        written for.
        """
        return any(watch.marker in line and self._armed(line) == watch.arms for line in self.watchers())

    def producer(self, watch: Watch) -> str:
        """The running process line this watch's gate resolves to, or empty when nothing is producing it.

        A gate is written two ways that look nothing alike. `ARM DONE <arm>` names the ARM; `pid:<n>`
        names the PROCESS. Every liveness test here crosses MARKERS, so two watchers gated on one running
        job in the two different notations read as two independent producers, and the queue that owes them
        both reads healthy. Resolving each gate to the process LINE is what makes them comparable.

        A numeric token is matched as a FIELD, never as a substring: a pid is four or five digits and
        those turn up inside paths, seeds and memory columns of unrelated rows.
        """
        token = watch.marker.removeprefix(DONE).removeprefix(PID).strip()
        as_field = token.isdigit()
        rows = (line for line in self.running.splitlines() if not self._mentions_watcher(line))
        return next((line for line in rows if (token in line.split() if as_field else token in line)), "")

    def collided(self) -> list[str]:
        """Every arm a live watcher will start behind a producer ANOTHER live watcher is also behind.

        `doubled` catches two watchers owing the SAME arm. This catches the opposite shape and the one
        that actually happened: two watchers owing DIFFERENT arms, both gated on one running job, which
        puts two arms on the card the moment that job exits. `after.sh` serialises within one watcher and
        knows nothing of its peers, so nothing downstream prevents it.
        """
        behind: dict[str, list[Watch]] = {}
        for watch in self.watches():
            if self.alive(watch) and watch.waiting() and (line := self.producer(watch)):
                behind.setdefault(line, []).append(watch)
        clashing = [group for group in behind.values() if len({watch.waiting()[0] for watch in group}) > 1]
        return sorted({watch.waiting()[0] for group in clashing for watch in group})

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

    def misarmed(self) -> list[str]:
        """Live watchers polling a file that does not exist while their producer is already running.

        MEASURED 2026-08-26. `after.sh` waits by GREPPING A FILE, so a path that will never appear and a
        path whose writer has not reached it are the same observation — the watcher prints its usual
        `waiting for` line either way, `alive` says one live process carries it, and nothing is doubled
        or stranded. Every verdict this check had was green while the queue was dead.

        THE CROSSING IS WITH THE PRODUCER, and that is what makes it safe to BLOCK on. A watcher armed
        before its producer starts is the normal case and its file is legitimately absent, so silence is
        correct there. But if the arm named in the marker is ON THE CARD and has still written nothing at
        that path, the path is wrong — a running arm writes its full log from its first line.

        The producer is looked for OUTSIDE the watcher lines, because every watcher carries its own
        marker in argv and would otherwise vouch for itself.
        """
        return [
            f"{watch.log} -> {watch.watched}"
            for watch in self.watches()
            if watch.watched
            and self.alive(watch)
            and not self.local(watch.watched).is_file()
            and self._producing(watch)
        ]

    def unlogged(self) -> list[str]:
        """Live watchers this check cannot see, because their log is not in the directory it reads.

        MEASURED 2026-08-26. The live chain owing `calib_dense_hold` and `ema_dense` wrote its watcher
        log to `experiments/logs/` while this check globs `logs/`, so it read eleven dead logs from a
        finished morning and reported `OK: every arm still owed is gated by exactly one live process`.
        Every arm it vouched for had already run, and the only queue in the tree was invisible to it.

        THIS IS THE MODULE'S OWN THESIS TURNED ON ITSELF. The whole point here is that a log and a
        process say different things and only the crossing is true — and until now a MISSING log was the
        one case that produced no crossing at all, so the check fell back to the logs and called them
        the queue. `stranded` reads a log with no process; this reads a process with no log.

        BLOCKING, because the verdict it corrupts is the green one. A stranded arm is announced loudly;
        a check that is merely looking in the wrong place says OK, which is the shape this repo has now
        paid for three times.

        GENERIC ONLY. A bespoke `after_<thing>.sh` takes no arm in argv, so `_armed` is empty for one and
        it would match no watch here and fire this BLOCKING alarm forever. Bespoke watchers are reported
        by `bespoke` instead, which does not pretend to know what they owe.
        """
        seen = {watch.arms for watch in self.watches()}
        return [
            " ".join(armed)
            for line in self.watchers()
            if self._script(line) == WATCHER and (armed := self._armed(line)) not in seen
        ]

    def _producing(self, watch: Watch) -> bool:
        """Whether the arm this watch waits on is itself live, read off any line that is not a watcher."""
        producer = watch.marker.removeprefix(DONE)
        return any(producer in line for line in self.running.splitlines() if not self._mentions_watcher(line))

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

        Two of these failures are opposite and both are silent: an arm NO live watcher will start, and an
        arm SEVERAL of them will. One wastes a card that is idle, the other oversubscribes a card that is
        not. The third is a watcher that is live, singular and pointed at nothing.

        UNLOGGED is reported FIRST because it is the only one that invalidates the others. The rest say
        this queue is wrong; that one says the queue being read is not the queue that is running, and a
        MISARMED or STRANDED verdict computed over the wrong directory is noise either way.
        """
        found = []
        if duplicated := self.duplicated():
            found.append(
                f"DUPLICATE WATCHER: {' '.join(duplicated)} — one bespoke watcher script is carried by more "
                "than one live process, and each will start the arm hardcoded in its body, so kill all but "
                "one (Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\", then Stop-Process)"
            )
        if unlogged := self.unlogged():
            found.append(
                f"UNLOGGED: {' '.join(unlogged)} — a live after.sh owes these and has no watcher log under "
                f"{self.directory}, so every other verdict here was reached without seeing it; re-run with "
                "--directory pointed at the log that chain actually writes"
            )
        if misarmed := self.misarmed():
            found.append(
                f"MISARMED: {' '.join(misarmed)} — the watched file does not exist while the arm named in "
                "the marker is running, so this watcher will wait forever; re-arm it on the real full log "
                "(arm full logs carry the SEED, `<arm>_s<seed>.full.log`)"
            )
        if collided := self.collided():
            found.append(
                f"COLLIDED: {' '.join(collided)} — these are owed by DIFFERENT live watchers gated on one "
                "running job, so they all start the moment it exits; kill all but one chain and re-queue "
                "the rest behind the arm that wins the card, not behind the job they were both watching"
            )
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
        bespoke = self.bespoke()
        if not watches:
            head = [f"no watcher logs under {self.directory}"]
            if not bespoke:
                return [*head, "OK: nothing is queued"]
            return [
                *head,
                *self.faults(),
                f"REPORT: {' '.join(bespoke)} live — bespoke watchers, each owing an arm named in its body "
                "and in no log, so what they owe cannot be read here; this is NOT an idle card",
            ]
        rows = [f"{len(watches)} watcher logs under {self.directory}"]
        if bespoke:
            rows.append(f"  (plus {' '.join(bespoke)} live — bespoke, owed arm not readable from a log)")
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
        index = QueuedArms._watcher_at(line)
        if index < 0:
            return ()
        tail = line.split()[index + 1 :]
        return tuple(token.strip("\"'") for token in tail if token.strip("\"'").endswith(ARM_SUFFIX))

    @staticmethod
    def _script(line: str) -> str:
        """The watcher script this process line runs, by basename, or empty when it is not a watcher."""
        index = QueuedArms._watcher_at(line)
        if index < 0:
            return ""
        return PurePosixPath(line.split()[index].strip("\"'").replace("\\", "/")).name

    def bespoke(self) -> list[str]:
        """Live watchers that are not the generic chain script, each with its live process count.

        A bespoke `after_<thing>.sh` hardcodes the arm it owes in its BODY rather than taking it in argv,
        so `waiting()` cannot be computed for one and the log-crossing every other verdict here rests on
        does not exist. That is a reason to REPORT it, never a reason to omit it: before this existed the
        omission was total and the check said `OK: nothing is queued` over a live queue.
        """
        scripts = [script for line in self.watchers() if (script := self._script(line)) and script != WATCHER]
        return [f"{script} x{scripts.count(script)}" for script in dict.fromkeys(scripts)]

    def duplicated(self) -> list[str]:
        """Bespoke watcher scripts carried by more than one live process — one arm, two launches.

        `doubled` reads arms out of chain logs and cannot see these, because a bespoke watcher names its
        arm nowhere a log can be read. Counting the PROCESSES per script catches the same failure by its
        other end, and it is the one that actually happened on 2026-09-11: a botched launch and its
        correction both survived, two live `after_holdclean.sh`, one arm, one card.
        """
        scripts = [script for line in self.watchers() if (script := self._script(line)) and script != WATCHER]
        return sorted({script for script in scripts if scripts.count(script) > 1})

    @staticmethod
    def local(watched: str) -> Path:
        """A path as `after.sh` recorded it, rewritten so THIS interpreter can open the same file.

        MEASURED 2026-08-26, and it made `misarmed` fire on a HEALTHY queue. Watchers are launched from
        git-bash, which writes an absolute path as `/d/personal_projects/…`; Windows Python resolves that to
        a `\\d\\personal_projects\\…` on the CURRENT drive, which does not exist — so `is_file()` is False for
        a file plainly sitting on disk. Every git-bash-launched watcher read as pointed at nothing, and
        `misarmed` BLOCKS and advises re-arming, which means the advice was to tear down a live chain. That
        is the shape `landed` was written for: a verdict whose action costs the card, about work that is fine.

        Only the `/<letter>/` prefix is rewritten, and a genuine POSIX path is left alone, so the rule adds
        no new way to be wrong on a machine where the recorded path was already openable.
        """
        head, _, tail = watched.lstrip("/").partition("/")
        drive = watched.startswith("/") and len(head) == 1 and head.isalpha()
        return Path(f"{head}:/{tail}") if drive else Path(watched)

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
