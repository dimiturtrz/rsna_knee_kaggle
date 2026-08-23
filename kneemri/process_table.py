"""This machine's process table, read whichever way this machine allows.

Two checks now cross something against the set of live processes — `queue_check`, which asks whether
every queued arm still has a watcher, and `card_check`, which asks whether the GPU is held by more than
one of our own jobs. Both need the FULL command line rather than an image name, and both have to work
from a Windows Python process and from git-bash, which do not agree on what is on PATH.

It lives at the package root because that is what it is: shared vocabulary two producers speak, not a
bucket of loose types. Neither check owns it and neither should import the other to reach it.

The table is read once into a value and passed INTO the crossings rather than read inside them, so each
crossing stays a pure function of what it was handed and its unit test needs no process to kill.
"""

import logging
import shutil
from dataclasses import dataclass

from kneemri.spawn import Spawn

logger = logging.getLogger(__name__)

IDS = 2
"""How many whole numbers a process row carries before its argv: its own id, then its parent's."""


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
        (
            "-NoProfile",
            "-Command",
            'Get-CimInstance Win32_Process | ForEach-Object { "$($_.ProcessId) $($_.ParentProcessId) '
            '$($_.CommandLine)" }',
        ),
    ),
)
"""Ordered, first non-empty wins — and the second one is why this is a list rather than a call.

MEASURED 2026-08-20: `ps -ef` lists the running arm correctly in the harness shell, and
`shutil.which("ps")` from the interpreter this check runs in returns None. The MSYS `ps` lives in
git-bash's `/usr/bin`, which is not on the PATH a Windows Python process inherits, so the check
degraded to "liveness cannot be crossed" and returned success — the exact silent pass it exists to
prevent, on the only machine it runs on.

`Get-CimInstance Win32_Process` is the Windows counterpart and is strictly better here: it carries the
FULL command line, which is what the crossings match on, where `tasklist` carries only an image name
and would match nothing at all.

THE TWO ARE NOT INTERCHANGEABLE FOR PIDS, which `card_check` found on 2026-08-23. MSYS `ps` reports
MSYS pids and every other tool on this machine — `nvidia-smi` included — reports Windows ones, so a
crossing that picks the wrong probe returns a clean, confident, entirely empty answer.
"""


@dataclass(frozen=True)
class ProcessTable:
    """One reading of this machine's process table, and the grammar for taking a row apart."""

    text: str

    @classmethod
    def read(cls, probes: tuple[Probe, ...] = PROBES) -> "ProcessTable":
        """The first probe that answers, or an empty table when none does.

        An unreadable table must not read as "nothing is running" — that turns every queued arm into a
        false strand and every busy card into a free one — so emptiness is preserved for the caller to
        report, rather than being smoothed into an answer.
        """
        for probe in probes:
            found = probe.read()
            if found:
                return cls(found)
        return cls("")

    def commands(self) -> dict[str, str]:
        """Each pid mapped to its argv, bookkeeping columns already dropped.

        Built once rather than searched per lookup, because the table is thousands of lines on this
        machine and the things crossed against it are single digits.
        """
        found: dict[str, str] = {}
        for line in self.text.splitlines():
            pid = self.parentage(line)[0]
            if pid and pid not in found:
                found[pid] = self.command(line)
        return found

    @staticmethod
    def parentage(line: str) -> tuple[str, str]:
        """This process line's own id and its parent's, or a pair of blanks when it carries neither.

        Read as THE FIRST TWO WHOLE-NUMBER TOKENS, which is the one thing both probes agree on: `ps -ef`
        prints them after the user and PowerShell is asked to print them first. A line they cannot be
        read from keeps the process — a watcher dropped for an unparseable id would be a strand a check
        invents, and a card holder dropped for one would be a job a check fails to see.
        """
        numbers = [token for token in line.split() if token.isdigit()]
        return (numbers[0], numbers[1]) if len(numbers) > 1 else ("", "")

    @staticmethod
    def command(line: str) -> str:
        """This process line with its bookkeeping columns dropped, leaving the argv a human reads.

        Both probes prepend fields nobody crossing a process table cares about — `ps -ef` a user, a cpu
        share and three time columns, PowerShell the two ids it was asked for — and a report that prints
        them spends its first thirty characters on numbers already shown in the row's own pid column.

        Cut AFTER THE PARENT ID rather than at a fixed offset, which is the one landmark both formats
        share and the same landmark `parentage` reads. A line with no parent id is returned whole,
        because a line this cannot parse is still worth showing.
        """
        tokens = line.split()
        seen = 0
        for position, token in enumerate(tokens):
            seen += token.isdigit()
            if seen == IDS:
                return " ".join(tokens[position + 1 :])
        return line.strip()
