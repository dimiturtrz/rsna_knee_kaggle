"""Is the GPU held by exactly one of our jobs?

`python -m kneemri.experiments.card_check`

MEASURED 2026-08-23: two identical `member_curve` scans ran concurrently on this card for twenty-five
minutes. The second was launched by `nohup` on top of a first that had NOT failed — the documented hole
where the launcher's own exit status says nothing about the job it detached.

THE COST WAS NOT THE WASTED CARD, IT WAS THE READING. Contention stretched one member's inference from
about three minutes to eleven, and that number was one step from being written down as a fact about the
member ("the DINOv2-base one is slow"). A harness fault wearing a measurement's clothes is the exact
class this project's instructions say to suspect before believing any result, and nothing here could
have caught it: `queue_check` guards WATCHERS, and both symmetric holes it blocks are about arms nobody
or everybody owes. Neither says anything about who holds the device.

IT ASKS THE DEVICE, NOT THE PROCESS TABLE, and that is the whole design. Deciding "is this a card job?"
from a command line means keeping a list of module names that take the GPU, which is wrong the first
time an arm is renamed and wrong in the other direction for `member_curve --device cpu`, which reads
exactly like a card job and takes no card at all. `nvidia-smi` reports the pids that actually hold the
device, and it reports them whoever started them.

WHICH IS ALSO HOW IT SEES THE OTHER PROJECT. Another repo on this machine trains on the same card by
arrangement, so "the card is busy" is not by itself a fault — the fault is TWO OF OURS. Each holder is
crossed against the process table to be named, and a holder that is not ours is reported as shared
rather than as a problem, because the lanes rule makes it not ours to kill.

THREE THINGS THE DRIVER WILL NOT TELL YOU HERE, all found on this check's first live run against a
5090 on the Windows display driver, and each of which had already been coded around wrongly:

`--query-compute-apps` IS NOT THE COMPUTE APPS. It answered with thirty-one processes including
`explorer.exe` and two copies of Firefox, because on this driver that query enumerates every client of
the device rather than the compute ones. The `Type` column of the plain table is what separates them,
`C` from `C+G`, and it exists in no CSV query — so this check parses the human table, which is the
opposite of the usual advice and is here the only source of the field that matters.

PER-PROCESS MEMORY IS `[N/A]`, for every process, in both the CSV query and the plain table. It is not
a permission problem and it is not fixable from here; the display driver does not account memory per
client. So this check reports the DEVICE total, which is answered correctly, and never pretends to a
per-process figure — a column of zeroes would have read as "nothing is really running".

AND THE DRIVER'S OWN `process_name` IS WRONG. It named our training process
`...rs\\User\\miniconda3\\python.exe` while the process table had the same pid as
`…\\rsna_knee_kaggle\\.venv-arm\\Scripts\\python.exe`. Had the ours/theirs split been decided on the
driver's name, this check would have called our own job the other project's — the precise misread it
exists to prevent. The pid is the only field taken from the driver; every name comes from the crossing.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from core import cli
from kneemri.process_table import PROBES, Probe, ProcessTable

logger = logging.getLogger(__name__)

DEVICE = Probe("nvidia-smi", ())
"""The plain, human-formatted report — parsed for its `Type` column, which no CSV query exposes."""

TOTAL = Probe("nvidia-smi", ("--query-gpu=memory.used,memory.total", "--format=csv,noheader"))
"""Device-wide memory, which IS answered here, standing in for the per-process figures that are not."""

COMPUTE = "C"
"""The `Type` of a process using the device to compute. A window manager is `G`, a browser `C+G`.

MATCHED EXACTLY, never as a prefix, and the difference is twenty-four processes. A first cut tested
`startswith`, which is true of `C+G` as well, so the check reported Explorer and two copies of Firefox
as holders of the card. The whole value of reading the plain table instead of a CSV query is this one
field, and reading it loosely throws away exactly what it was parsed for.
"""

REPO = Path(__file__).resolve().parents[2]
"""This checkout's root, derived from the module's own location rather than written down.

A hardcoded path would be the machine-specific-path defect the project instructions forbid, and it
would also be the WRONG check on a second checkout: what makes a holder ours is that it runs from THIS
tree, not that it runs from a tree with a particular name.
"""


@dataclass(frozen=True)
class DeviceReport:
    """The driver's plain report, parsed for the one field no CSV query exposes."""

    text: str

    def compute_pids(self) -> list[str]:
        """The pids computing on the device, read off the plain report's process table.

        Rows are `| GPU GI CI PID Type Process-name Memory |`, so the pid is the last numeric field
        BEFORE the type and the type is the field after it. Matched that way round rather than by
        column index, because the leading `GI`/`CI` fields are `N/A` on a card without MIG and present
        on one with it, which moves every index by two.
        """
        found: list[str] = []
        for row in self.text.splitlines():
            fields = row.strip("| ").split()
            for position, field in enumerate(fields[:-1]):
                if field.isdigit() and fields[position + 1] == COMPUTE:
                    found.append(field)
                    break
        return found


PREVIEW_WIDTH = 110
"""How much of one holder's argv a row shows. A terminal row, not a policy — what it is SPENT on is."""

ELISION = " ... "
"""What stands where the shared defaults were, so a reader can see the row is not the whole command."""


@dataclass(frozen=True)
class Holder:
    """One process computing on the device, named by the process table rather than by the driver."""

    pid: str
    command: str

    def ours(self, repo: Path = REPO) -> bool:
        """Whether this holder runs out of this checkout.

        A holder the process table could not name is NOT ours — it is reported as unnamed and counted
        as shared. Claiming an unidentifiable process would let the check block our own launch because
        of someone else's job, which is the failure that wastes the most wall-clock: a refusal nobody
        can act on.
        """
        return self.normalised(str(repo)) in self.normalised(self.command)

    def line(self) -> str:
        """This holder as one row, elided because a full argv here is a screen of flags."""
        who = "OURS " if self.ours() else "other"
        return f"  {who} pid {self.pid:>7}  {self.elided()}"

    def elided(self, width: int = PREVIEW_WIDTH) -> str:
        """The command at preview width with the MIDDLE dropped — never the tail.

        BOTH ENDS ARE LOAD-BEARING AND THE MIDDLE IS NOT. Every arm here runs `$RECIPE $FIT`: the
        shared defaults first, the arm's own overrides last. A head-truncated preview therefore shows
        precisely the flags that are being overridden and never the ones that were moved —
        `small_batch6_ff` ran `--fold -1 --ema` and the preview displayed `--fold 0`. Read at face
        value that says the one deliberate flag of the arm did not land, which indicts a correct
        95-minute run mid-flight; it resolved the other way only by going around this tool entirely.
        The head still names the entrypoint, which is what says whose job this is, so the part that can
        be spent is the defaults nobody reads.
        """
        if not self.command:
            return "(not in the process table)"
        if len(self.command) <= width:
            return self.command
        kept = width - len(ELISION)
        head = kept // 2
        return self.command[:head] + ELISION + self.command[head - kept :]

    @staticmethod
    def normalised(text: str) -> str:
        """One spelling for a path that three tools disagree about.

        A command line reaches here from `ps -ef` with forward slashes, from `Get-CimInstance` with
        backslashes, and from either with whatever case the launcher happened to type. Matching a repo
        root against any of those without normalising is a check that passes on this machine and
        silently stops recognising our own jobs on the next one.
        """
        return text.replace("\\", "/").lower()


class CardHolders:
    """Who computes on the GPU, crossed from the driver's pids against a process table."""

    def __init__(self, report: DeviceReport, table: ProcessTable) -> None:
        self.report = report
        self.table = table

    @classmethod
    def crossed(cls, report: DeviceReport, probes: tuple[Probe, ...] = PROBES) -> "CardHolders":
        """The driver's answer crossed against whichever probe names the most of its pids.

        THE TWO PID NAMESPACES ARE THE REASON THIS EXISTS. `nvidia-smi` reports Windows pids and the
        MSYS `ps -ef` reports MSYS ones, so crossing the driver against the wrong probe yields a clean,
        confident, entirely empty answer. Rather than branch on the operating system, each probe is
        scored by how many of the driver's own pids it accounts for and the best one wins — which is
        self-validating, needs no platform test, and is right on a machine where only one probe exists.

        A probe that answers nothing at all is skipped, so an unnameable holder is reported as unnamed
        rather than making the whole crossing unavailable.
        """
        tables = [ProcessTable(found) for found in (probe.read() for probe in probes) if found]
        if not tables:
            return cls(report, ProcessTable(""))
        return max((cls(report, table) for table in tables), key=cls.named)

    def holders(self) -> list[Holder]:
        """Every compute process on the device, ours sorted first so the interesting rows read at top."""
        commands = self.table.commands()
        found = [Holder(pid, commands.get(pid, "")) for pid in self.report.compute_pids()]
        return sorted(found, key=lambda holder: not holder.ours())

    def named(self) -> int:
        """How many holders the table could name — the score `crossed` picks a probe by."""
        return sum(1 for holder in self.holders() if holder.command)

    def ours(self) -> list[Holder]:
        """The holders that run out of this checkout — the only ones this check is entitled to count."""
        return [holder for holder in self.holders() if holder.ours()]

    def faults(self) -> list[str]:
        """Every way the card is wrong to launch onto, or empty when it is not.

        ONLY OUR OWN JOBS COUNT. A second project trains on this card by arrangement, so a busy card is
        the normal state and blocking on it would make this check something to be ignored, which is
        worse than not having it. Two of ours is the fault, because that is the one nobody arranged.
        """
        mine = self.ours()
        if len(mine) > 1:
            return [
                f"OVERSUBSCRIBED: {len(mine)} of our own processes compute on this card "
                f"(pids {' '.join(holder.pid for holder in mine)}) — one was almost certainly launched "
                "on top of a job that had not failed, and they will each read slower than either would "
                "alone; kill all but one before trusting any timing from them"
            ]
        return []

    def lines(self, memory: str = "") -> list[str]:
        """Who holds the card, then the verdict — verdict last so a tail sees it.

        Memory is reported for the DEVICE and never per holder: this driver answers `[N/A]` for every
        process, and a column of zeroes would read as an idle card rather than as an unanswered
        question.
        """
        holders = self.holders()
        used = f", device at {memory.strip()}" if memory.strip() else ""
        if not holders:
            return [f"no process computes on this card{used}", "OK: the card is free"]
        rows = [f"{len(holders)} process(es) compute on this card{used}"]
        rows += [holder.line() for holder in holders]
        shared = len(holders) - len(self.ours())
        if shared:
            rows.append(f"  ({shared} not ours — shared by arrangement, not ours to kill)")
        return [*rows, *(self.faults() or ["OK: at most one of our jobs holds this card"])]


def main() -> None:
    """Report a card held by more than one of our own jobs, before a launch adds another.

    EXITS NON-ZERO ON A FAULT AND ON AN UNREADABLE DRIVER, under different codes, for the reason
    `queue_check` states at the same place: a warning that leaves the status at 0 reads, in a log tail,
    exactly like a pass.
    """
    command = cli.Command(__doc__, logger)
    command.parsed()
    report = DEVICE.read()
    if not report.strip():
        command.unknown("nvidia-smi could not be read, so the card's holders cannot be crossed")
    card = CardHolders.crossed(DeviceReport(report), PROBES)
    command.report(card.lines(TOTAL.read()))
    command.conclude(card.faults(), "fault(s) on the card")


if __name__ == "__main__":
    main()
