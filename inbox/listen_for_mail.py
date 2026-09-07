#!/usr/bin/env python3
"""Mailbox listener — arm as a harness background task, exits on mail.

One script, three boxes, resolved BY NAME under this script's own directory:

    inbox/main/         letters TO the main agent
    inbox/researcher/   letters TO the researcher
    inbox/mastermind/   letters TO the coordinator

`--box <name>` is REQUIRED and has no default: the previous listeners resolved a box from the
directory the script happened to sit in, so a copy silently watched a different side's mail and a
move relocated a mailbox with an arm in flight. A name cannot do either. A box is armed by its OWNER
— arming somebody else's box archives their unread mail into `read/` where they never see it.

On arrival: prints each letter's body, archives it to `<box>/read/`, and exits, so the harness
delivers the output as a background-task completion notification. Re-arm FIRST on every notification,
then read, then work.

Stdlib-only polling: filesystem-event listeners (watchdog) are a dependency this repo does not
otherwise need, and events were unreliable enough in the fpga project that its listener drained on a
5s poll regardless.

Usage: python inbox/listen_for_mail.py --box main|researcher|mastermind [timeout_seconds]
Arm with a harness timeout slightly above the internal one. One listener per box: a second box is a
second background task, not a flag.
"""

import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents import Registry, Status, utc_now  # noqa: E402  (the path above is what makes it importable)

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

POLL_SECONDS = 2
HEARTBEAT_STALE_SECONDS = 30
DEFAULT_TIMEOUT_SECONDS = 82800  # arm the background task with 86400

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOXES = ("main", "researcher", "mastermind")

EXTENSIONS = (".txt", ".md")
NOT_LETTERS = ("README.md",)
"""A box documents itself in the box; that file is not mail and archiving it empties the documentation."""

INBOX = ""
READ_DIR = ""
PID_FILE = ""
HEARTBEAT_FILE = ""


def _select_box(box: str) -> None:
    """Point the module's paths at one box. Per-box state files, so two boxes can be armed at once."""
    global INBOX, READ_DIR, PID_FILE, HEARTBEAT_FILE
    INBOX = os.path.join(SCRIPT_DIR, box)
    READ_DIR = os.path.join(INBOX, "read")
    PID_FILE = os.path.join(INBOX, ".listener_pid")
    HEARTBEAT_FILE = os.path.join(INBOX, ".listener_heartbeat")


def _parsed(argv: list[str]) -> tuple[str, int]:
    """`--box <name>` anywhere, one optional positional timeout — argparse is not worth a dependency."""
    box, timeout, rest = "", DEFAULT_TIMEOUT_SECONDS, []
    index = 0
    while index < len(argv):
        if argv[index] == "--box":
            if index + 1 >= len(argv) or argv[index + 1] not in BOXES:
                print(f"--box takes one of {', '.join(BOXES)}", file=sys.stderr)
                sys.exit(2)
            box = argv[index + 1]
            index += 2
            continue
        rest.append(argv[index])
        index += 1
    if not box:
        print(f"--box is required, one of {', '.join(BOXES)} — a box is never inferred", file=sys.stderr)
        sys.exit(2)
    if rest:
        try:
            timeout = int(rest[0])
        except ValueError:
            print(f"invalid timeout: {rest[0]}", file=sys.stderr)
            sys.exit(2)
    return box, timeout


def _refuse_if_inactive(box: str) -> None:
    """A lane switched off in `inbox/agents.csv` is not armed, and its mail is left where it is.

    Checked on arm AND on every poll, because switching a lane off has to work on one that is already
    running. Exit 2 rather than a silent return: the harness records a status, and 'not arming' must
    never look like 'armed and quiet'. Unread letters are never archived while inactive — off means
    silenced, never lost, and that is the whole safety property of the switch.
    """
    if Registry.read().active(box):
        return
    note = Registry.read().agent(box).note
    print(f"box {box} is INACTIVE in inbox/agents.csv ({note}) — not arming", file=sys.stderr)
    sys.exit(2)


def _touch_heartbeat() -> None:
    try:
        with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
            f.write(str(time.time()))
    except OSError:
        pass


def _heartbeat_is_fresh() -> bool:
    try:
        return (time.time() - os.path.getmtime(HEARTBEAT_FILE)) < HEARTBEAT_STALE_SECONDS
    except OSError:
        return False


def _clear_state_files() -> None:
    for path in (PID_FILE, HEARTBEAT_FILE):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def _kill_pid(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        return
    try:
        os.kill(pid, 15)
        time.sleep(2)
        os.kill(pid, 0)
        os.kill(pid, 9)
    except OSError:
        pass


def _kill_existing_listener() -> None:
    """Last-writer-wins: a zombie listener from a dead session would race us to archive mail before the
    agent sees it. Only kill when the heartbeat is fresh — a stale one means the PID may have been
    recycled by an unrelated process (Windows PID reuse)."""
    try:
        with open(PID_FILE, encoding="utf-8") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        _clear_state_files()
        return
    if pid != os.getpid() and _heartbeat_is_fresh():
        _kill_pid(pid)
    _clear_state_files()


def _record(box: str, delivered: int) -> None:
    """This listener's own runtime line — the half of `agents.py --list` a human cannot type.

    One writer per file (this box's listener), so no lock is needed. `delivered` accumulates across
    arms: a lane that has delivered nothing all day is a different picture from one that has.
    """
    was = Status.read(box)
    Status(
        pid=os.getpid(),
        armed_utc=was.armed_utc if was is not None and delivered else utc_now(),
        last_delivered_utc=utc_now() if delivered else (was.last_delivered_utc if was is not None else ""),
        delivered=(was.delivered if was is not None else 0) + delivered,
    ).written(box)


def _letters() -> list[str]:
    if not os.path.isdir(INBOX):
        return []
    return sorted(
        f
        for f in os.listdir(INBOX)
        if f.endswith(EXTENSIONS) and f not in NOT_LETTERS and os.path.isfile(os.path.join(INBOX, f))
    )


def _archive(basename: str) -> None:
    src = os.path.join(INBOX, basename)
    dst = os.path.join(READ_DIR, basename)
    for _ in range(50):  # 5s of retries rides out Windows file-lock races
        try:
            shutil.move(src, dst)
            return
        except PermissionError:
            time.sleep(0.1)
        except FileNotFoundError:
            return  # another listener already filed it
    print(f"WARNING: could not archive {basename}", file=sys.stderr)


def _deliver(names: list[str]) -> None:
    box = os.path.basename(INBOX)
    for basename in names:
        path = os.path.join(INBOX, basename)
        try:
            with open(path, encoding="utf-8") as f:
                body = f.read()
        except OSError:
            body = ""
        _archive(basename)
        print(f"=== inbox/{box}/{basename} ===")
        print(body)
    _record(box, delivered=len(names))
    print(f"MAIL:{len(names)}:{','.join(names)}")
    print("---")
    print("Listener exited on mail. Re-arm now:")
    print(f"  python inbox/listen_for_mail.py --box {box} 86000")


def main() -> None:
    box, timeout = _parsed(sys.argv[1:])
    _select_box(box)
    _refuse_if_inactive(box)

    os.makedirs(READ_DIR, exist_ok=True)
    _kill_existing_listener()

    try:
        with open(PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except OSError as e:
        print(f"WARNING: pid write failed: {e}", file=sys.stderr)
    _touch_heartbeat()
    _record(box, delivered=0)
    start = time.time()

    try:
        pending = _letters()  # drain mail that predates this arm
        if pending:
            _deliver(pending)
            return
        while True:
            time.sleep(POLL_SECONDS)
            _refuse_if_inactive(box)
            _touch_heartbeat()
            if not os.path.isdir(INBOX):
                print("ERROR: box directory was deleted", file=sys.stderr)
                sys.exit(1)
            pending = _letters()
            if pending:
                _deliver(pending)
                return
            if time.time() - start > timeout:
                print(f"TIMEOUT: no mail after {timeout}s. Re-arm now:")
                print(f"  python inbox/listen_for_mail.py --box {box} {timeout}")
                return
    finally:
        _clear_state_files()


if __name__ == "__main__":
    main()
