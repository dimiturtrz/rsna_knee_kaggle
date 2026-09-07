"""Did the kernel run the entry it was named for? `python -m kneemri.kaggle.run_check --log <path>`.

WHY THIS IS A CHECK AND NOT A GLANCE. `KernelRun` DROPS a member whose encoder it cannot mount and keeps
going, because a seven-member entry that becomes a six-member entry is still a valid submission and dying
on the last member of a nine-hour run would be worse. It says so — `ENTRY IDENTITY CHANGED` — and then it
writes a perfectly good `submission.csv`. So the failure is SILENT in every place a human looks: the
kernel reads COMPLETE, the output file exists, the submission scores, and the number lands on the board
under the name of an entry that was never assembled.

WHAT IT COST. `farseat_v1` (ref 55758082, 2026-08-25) shipped six of its seven named members, was `dark6`
seat-for-seat, scored `dark6`'s exact 0.937, and that flat read was written down as evidence that a
seventh seat is inert even ACROSS architecture families — closing the cross-family ensemble-diversity
program and `knee-avzz`'s under-pricing argument with it. It closed nothing; the seventh seat was never in
the entry. Five days passed. `mount_check` now asks the question that would have caught it before the run
(does the encoder dataset publish the path the mount names), and this asks the one that catches it after:
did the run that produced this submission.csv load every member the entry claims. See knee-nbk3.

READ IT BEFORE THE SLOT, NOT AFTER. `kaggle kernels output` serves the LATEST version's log, so a row
written days later cannot be re-checked — the evidence is gone. That is why this is a pre-submission gate
rather than a rule inside `board_check`: the only moment the log is both available and still able to
change a decision is between the kernel completing and the slot being spent.
"""

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from core import cli

logger = logging.getLogger(__name__)

IDENTITY_MARK = "ENTRY IDENTITY CHANGED"
"""What `KernelRun` prints when it served fewer members than it was named for."""

FALLBACK_MARK = "shipping our six alone"
"""What a two-producer kernel prints when the FOREIGN side did not run.

`kneemri-bend` wraps mattiaangeli's program so that a crash there leaves the champion's own table in
place — correct engineering, and a WRONG entry: what would then be submitted is the champion set under
a new name, which `submit_check` compares by member SET and would wave through as new. The fallback
exists so the kernel survives; this refusal exists so the slot is never spent on what it produced."""

LOADED = re.compile(r"loaded and fingerprint-checked (\d+) of (\d+) members")
"""The run's own census of its seats, printed once, before a single study is decoded."""

DROPPED = re.compile(r"member (\S+) \((\S+)\) would not load")
"""One line per seat that could not be assembled, naming the member and the encoder it wanted."""

STREAM = "data"
"""Kaggle serves a kernel log as JSON records; this is the field holding the line itself."""


@dataclass(frozen=True)
class KernelLog:
    """A completed kernel's log, reduced to the three things that decide whether its output is the entry.

    Parsed from Kaggle's JSON record form when it is one and from plain text when it is not, because a log
    pasted into a file by hand is the same evidence and refusing it would only teach the reader to skip
    the check.
    """

    lines: tuple[str, ...]

    @classmethod
    def of(cls, text: str) -> "KernelLog":
        """The log's lines, whether Kaggle's JSON records or the plain text of the same run."""
        try:
            records = json.loads(text)
        except json.JSONDecodeError:
            return cls(tuple(text.splitlines()))
        if not isinstance(records, list):
            return cls(tuple(text.splitlines()))
        return cls(tuple(str(record.get(STREAM, "")).rstrip() for record in records if isinstance(record, dict)))

    @classmethod
    def read(cls, path: Path) -> "KernelLog":
        """The log on disk, as `kaggle kernels output -p <dir>` leaves it."""
        return cls.of(path.read_text(encoding="utf-8", errors="replace"))

    def seats(self) -> tuple[int, int] | None:
        """Members loaded and members named, or None when the run never got as far as saying."""
        for line in self.lines:
            found = LOADED.search(line)
            if found:
                return int(found.group(1)), int(found.group(2))
        return None

    def dropped(self) -> list[str]:
        """Every seat the run could not assemble, as `<member> wanted <encoder>`."""
        found = (DROPPED.search(line) for line in self.lines)
        return [f"{drop.group(1)} wanted {drop.group(2)}" for drop in found if drop]

    def changed(self) -> bool:
        """Whether the run announced that it served an entry other than the one it was named for."""
        return any(IDENTITY_MARK in line for line in self.lines)

    def fell_back(self) -> bool:
        """Whether a two-producer run shipped one producer's table because the other did not arrive."""
        return any(FALLBACK_MARK in line for line in self.lines)

    def refusals(self) -> list[str]:
        """Every reason this run's output must not be submitted under the entry's name; empty when whole."""
        found = [f"the run DROPPED a seat: {drop}" for drop in self.dropped()]
        if self.fell_back():
            found.append(
                f"the run printed {FALLBACK_MARK!r} - the foreign producer never ran, so this table is the "
                f"champion set under another entry's name and the slot would buy a re-ship"
            )
        if self.changed():
            found.append(
                f"the run printed {IDENTITY_MARK!r} - its submission.csv is a smaller ensemble than the "
                f"entry names, and scoring it would record the smaller one under the larger one's name"
            )
        seats = self.seats()
        if seats is None:
            found.append("the run never printed its member census, so nothing here can say what it served")
        elif seats[0] != seats[1]:
            found.append(f"the run loaded {seats[0]} of {seats[1]} members")
        return found


def main() -> None:
    """Refuse a kernel run whose output is a different entry from the one its kernel is named for."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--log", type=Path, required=True, help="the kernel log to read")
    arguments = command.parsed()

    log = KernelLog.read(arguments.log)
    seats = log.seats()
    logger.info(
        "%s: %s",
        arguments.log,
        f"loaded {seats[0]} of {seats[1]} members" if seats else "no member census printed",
    )
    refusals = log.refusals()
    for refusal in refusals:
        logger.error("%s", refusal)
    if refusals:
        sys.exit(1)


if __name__ == "__main__":
    main()
