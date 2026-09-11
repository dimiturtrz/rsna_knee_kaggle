#!/usr/bin/env python3
"""The agent registry: which lanes are switched ON, and which of those are actually alive.

    python inbox/agents.py --list
    python inbox/agents.py --box researcher --off "auditing offline until the corpus lands"
    python inbox/agents.py --box researcher --on
    python inbox/agents.py --check researcher

TWO FILES, BECAUSE CONFIG AND RUNTIME TRUTH MUST NOT SHARE A HOME. `inbox/agents.csv` is what a human
wrote: one row per box, `active` 1 or 0, git-tracked, and the toggle below is its only programmatic
writer. `inbox/<box>/.status` is what that box's OWN listener wrote when it armed: gitignored, one
JSON line, exactly one writer, so nothing here needs a lock.

The failure this exists to make impossible is the dead-watcher one — a lane that LOOKS armed while
delivering nothing. A flag a human typed can never be the only evidence a lane is alive, so `--list`
JOINS the two and flags both lies: `active=1` with no live listener reads STALE, and `active=0` with
letters sitting in the box reads OFF with the count, because off must mean silenced and never lost.

LIVENESS IS THE HEARTBEAT, NOT THE ARM RECORD. `.status` is written when a listener ARMS and when it
DELIVERS; a listener that died in between leaves it behind saying `armed`, and reading its mtime is
how this file passed a lane whose listener had been dead for 55 minutes with two letters undelivered
(2026-09-11). The listener already maintains the right signal and deletes it on exit: it touches
`.listener_heartbeat` every poll, so a missing file is proof of death and a 30-second mtime bound is
a pulse. `Status.alive()` reads that; `Status.stale()` is a day-scale backstop for the wording only.

WHAT THE SWITCH DOES NOT DO. Switching a box off silences its MAIL. It does not stop that side's
wakeup loop, which is the owner's `/loop` control, and it does not let one side arm or disarm
another's listener — a box is still armed by its owner, one listener per box.
"""

import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOXES = ("main", "researcher", "mastermind")
REGISTRY = os.path.join(SCRIPT_DIR, "agents.csv")
STATUS_NAME = ".status"
HEARTBEAT_NAME = ".listener_heartbeat"

FIELDS = ("box", "role", "active", "note", "updated_utc")
ACTIVE, INACTIVE = "1", "0"
STALE_SECONDS = 24 * 60 * 60
HEARTBEAT_STALE_SECONDS = 30
LETTERS = (".txt", ".md")
NOT_LETTERS = ("README.md",)

_SEPARATOR = ","
_QUOTE = '"'


def utc_now() -> str:
    """The stamp both files carry, in the one format every letter name already uses."""
    return time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())


def _split(line: str) -> list[str]:
    """One CSV row, honouring quotes so a role or a reason may hold a comma.

    Hand-rolled rather than `csv`, for the same reason the listener hand-rolls its argument parse:
    this file is imported by a stdlib-only listener that must not grow a reason to fail on import.
    """
    cells, cell, quoted = [], "", False
    for character in line:
        if character == _QUOTE:
            quoted = not quoted
        elif character == _SEPARATOR and not quoted:
            cells.append(cell)
            cell = ""
        else:
            cell += character
    cells.append(cell)
    return cells


def _joined(cells: list[str]) -> str:
    """The inverse of `_split`: quote a cell only when it would otherwise change the row's shape."""
    return _SEPARATOR.join(f"{_QUOTE}{cell}{_QUOTE}" if _SEPARATOR in cell else cell for cell in cells)


class Agent:
    """One lane's config row: what it is for, and whether its mail is switched on."""

    def __init__(self, box: str, role: str, active: bool, note: str, updated_utc: str) -> None:
        self.box = box
        self.role = role
        self.active = active
        self.note = note
        self.updated_utc = updated_utc

    @classmethod
    def parsed(cls, line: str) -> "Agent":
        """A registry row as written; a short row is a corrupt file, not a default."""
        cells = _split(line)
        if len(cells) != len(FIELDS):
            message = f"registry row {line!r} holds {len(cells)} fields; the columns are {FIELDS}"
            raise ValueError(message)
        box, role, active, note, updated = cells
        return cls(box=box, role=role, active=active.strip() == ACTIVE, note=note, updated_utc=updated)

    def row(self) -> str:
        """The row this agent writes back, so a toggle round-trips through `parsed`."""
        return _joined([self.box, self.role, ACTIVE if self.active else INACTIVE, self.note, self.updated_utc])

    def switched(self, active: bool, reason: str) -> "Agent":
        """The same lane with its switch moved and the reason recorded beside the stamp."""
        return Agent(box=self.box, role=self.role, active=active, note=reason, updated_utc=utc_now())


class Status:
    """What a box's own listener last wrote about itself — the half a human cannot type."""

    def __init__(self, pid: int, armed_utc: str, last_delivered_utc: str, delivered: int) -> None:
        self.pid = pid
        self.armed_utc = armed_utc
        self.last_delivered_utc = last_delivered_utc
        self.delivered = delivered

    @classmethod
    def path(cls, box: str) -> str:
        """Where that box's listener writes, one file per box so there is one writer per file."""
        return os.path.join(SCRIPT_DIR, box, STATUS_NAME)

    @classmethod
    def read(cls, box: str) -> "Status | None":
        """The last arm's record, or None where no listener has ever armed this box."""
        try:
            with open(cls.path(box), encoding="utf-8") as handle:
                held = json.loads(handle.read())
        except (OSError, ValueError):
            return None
        return cls(
            pid=int(held.get("pid", 0)),
            armed_utc=str(held.get("armed_utc", "")),
            last_delivered_utc=str(held.get("last_delivered_utc", "")),
            delivered=int(held.get("delivered", 0)),
        )

    def written(self, box: str) -> None:
        """Record this arm. Written by the listener alone; every reader here treats it as evidence."""
        held = {
            "pid": self.pid,
            "armed_utc": self.armed_utc,
            "last_delivered_utc": self.last_delivered_utc,
            "delivered": self.delivered,
        }
        with open(self.path(box), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(held))

    @classmethod
    def stale(cls, box: str, now: float) -> bool:
        """Whether that box's arm is too old to be believed AT ALL — a day-scale backstop, never
        liveness. `.status` is written by `_record()` on arm and on delivery, NOT on every poll, so
        its mtime says when the listener last had something to say, not whether it is still there.
        This method used to be the only liveness test and it claimed otherwise in its own docstring;
        that is how a listener dead for 55 minutes read green. `alive()` is the liveness test."""
        return _age(cls.path(box), now) > STALE_SECONDS

    @classmethod
    def heartbeat(cls, box: str) -> str:
        """Where that box's listener touches on every poll — the only file whose mtime tracks a pulse."""
        return os.path.join(SCRIPT_DIR, box, HEARTBEAT_NAME)

    @classmethod
    def alive(cls, box: str, now: float) -> bool:
        """Whether a listener is polling this box RIGHT NOW, on the evidence the listener maintains.

        `listen_for_mail.py` touches `.listener_heartbeat` every `POLL_SECONDS` (2) and DELETES it on
        exit, so the file's absence is positive proof of death and its mtime bounds the pulse at
        `HEARTBEAT_STALE_SECONDS` (30) rather than at a day. Both constants mirror the listener's;
        they are duplicated rather than imported because the listener imports THIS module.

        A listener too old to write a heartbeat reads dead here, which is the safe direction: a lane
        wrongly flagged STALE costs a re-arm, a lane wrongly flagged live costs every letter in it.
        """
        return _age(cls.heartbeat(box), now) <= HEARTBEAT_STALE_SECONDS


def unread(box: str) -> int:
    """Letters sitting in a box and not yet delivered — what an OFF lane is holding for its owner."""
    directory = os.path.join(SCRIPT_DIR, box)
    if not os.path.isdir(directory):
        return 0
    return len(
        [
            name
            for name in os.listdir(directory)
            if name.endswith(LETTERS) and name not in NOT_LETTERS and os.path.isfile(os.path.join(directory, name))
        ]
    )


class Registry:
    """Every lane's config row, read and written as one file so a toggle cannot half-apply."""

    def __init__(self, agents: list[Agent]) -> None:
        self.agents = agents

    @classmethod
    def read(cls, path: str | None = None) -> "Registry":
        """The registry as written. A missing file is a refusal, never an implied all-active.

        The default is resolved HERE rather than in the signature so the constant can be pointed
        elsewhere for a test: a default bound at definition time would make every test of the
        listener's refusal path read the live registry and silence a real lane to exercise it.
        """
        with open(path or REGISTRY, encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.read().splitlines() if line.strip()]
        header, *body = lines
        if _split(header) != list(FIELDS):
            message = f"{path} heads with {header!r}; the columns are {FIELDS}"
            raise ValueError(message)
        return cls([Agent.parsed(line) for line in body])

    def agent(self, box: str) -> Agent:
        """One lane by name; an unknown box is a typo that must not read as an inactive lane."""
        for agent in self.agents:
            if agent.box == box:
                return agent
        message = f"no lane named {box!r}; the registry holds {', '.join(one.box for one in self.agents)}"
        raise ValueError(message)

    def active(self, box: str) -> bool:
        """Whether that lane's mail is switched on — the one question the listener asks."""
        return self.agent(box).active

    def toggled(self, box: str, active: bool, reason: str) -> "Registry":
        """The registry with one lane's switch moved, every other row byte-identical."""
        moved = self.agent(box).switched(active, reason)
        return Registry([moved if one.box == box else one for one in self.agents])

    def written(self, path: str | None = None) -> None:
        """Atomically, `.part` then replace — a half-written registry would disarm every lane at once."""
        path = path or REGISTRY
        part = f"{path}.part"
        with open(part, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(_joined(list(FIELDS)) + "\n")
            for agent in self.agents:
                handle.write(agent.row() + "\n")
        os.replace(part, path)

    def table(self, now: float) -> list[str]:
        """The JOINED view: config beside the runtime evidence, with both lies named.

        A lane flagged active with no listener POLLING it is the dead-watcher failure wearing a green
        flag, and liveness is `alive()` — the heartbeat — never the arm record. `stale()` only sharpens
        the complaint: an arm over a day old says the lane has been dark for a long time, a fresh one
        that stopped polling says it died since. A lane switched off with letters in the box is fine —
        but the count has to be visible, because that mail is waiting on a human to switch it back on.
        """
        lines = []
        for agent in self.agents:
            status = Status.read(agent.box)
            waiting = unread(agent.box)
            armed = status.armed_utc if status is not None else "never"
            flag = ""
            if agent.active and not Status.alive(agent.box, now):
                since = "; last arm over a day old" if Status.stale(agent.box, now) else f"; armed {armed}, not polling"
                flag = f"  STALE — flagged active, no live listener{since}"
            if not agent.active and waiting:
                flag = f"  OFF, {waiting} letters waiting"
            delivered = status.delivered if status is not None else 0
            lines.append(
                f"{agent.box:11s} active={int(agent.active)} armed={armed:17s} "
                f"delivered={delivered:<4d} unread={waiting:<3d} {agent.note or agent.role}{flag}"
            )
        return lines


def _age(path: str, now: float) -> float:
    """Seconds since a status file was last touched; a missing file is infinitely old, not fresh."""
    try:
        return now - os.path.getmtime(path)
    except OSError:
        return float("inf")


def _parsed(argv: list[str]) -> dict[str, str]:
    """`--list`, `--check <box>`, or `--box <x>` with `--on` / `--off <reason>`."""
    parsed: dict[str, str] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        takes_value = token in ("--box", "--check", "--off")
        if takes_value and index + 1 < len(argv):
            parsed[token.removeprefix("--")] = argv[index + 1]
            index += 2
            continue
        parsed[token.removeprefix("--")] = ""
        index += 1
    return parsed


def main() -> None:
    """Read the registry, or move one switch in it."""
    arguments = _parsed(sys.argv[1:])
    registry = Registry.read()

    if "check" in arguments:
        sys.exit(0 if registry.active(arguments["check"]) else 1)

    if "box" in arguments and ("on" in arguments or "off" in arguments):
        active = "on" in arguments
        reason = arguments.get("off", "")
        registry.toggled(arguments["box"], active, reason).written()
        print(f"{arguments['box']} is now {'ACTIVE' if active else 'INACTIVE'}{f' — {reason}' if reason else ''}")
        return

    for line in registry.table(time.time()):
        print(line)


if __name__ == "__main__":
    main()
