#!/usr/bin/env python3
"""Inbox listener — arm as a harness background task, exits on mail.

Watches Kimi's mailbox at .agent-runtime/inbox/ or Claude's at .claude/inbox/.
On arrival: prints each letter's body, archives it to the sibling read/
directory, and exits so the harness delivers the output as a background-task
completion notification.

Re-arm FIRST on every notification, then read, then work.

Stdlib-only polling: filesystem-event listeners (watchdog) are a dependency
this repo does not otherwise need, and events were unreliable enough in the
fpga project that its listener drained on a 5s poll regardless.

Usage: python .agent-runtime/listen_for_mail.py --name kimi|claude <timeout_seconds>
Arm with a harness timeout slightly above the internal one.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

POLL_SECONDS = 2
HEARTBEAT_STALE_SECONDS = 30

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _derive_paths(name):
    """Runtime mailboxes: Kimi at .agent-runtime/inbox/, Claude at .claude/inbox/."""
    if name == "claude":
        agent_dir = os.path.join(os.path.dirname(SCRIPT_DIR), ".claude")
        inbox_dir = os.path.join(agent_dir, "inbox")
    else:
        agent_dir = SCRIPT_DIR
        inbox_dir = os.path.join(agent_dir, "inbox")
    read_dir = os.path.join(inbox_dir, "read")
    pid_file = os.path.join(inbox_dir, f".listener_pid.{name}")
    heartbeat_file = os.path.join(inbox_dir, f".listener_heartbeat.{name}")
    return inbox_dir, read_dir, pid_file, heartbeat_file


EXTENSIONS = (".txt", ".md")


def _touch_heartbeat(heartbeat_file):
    try:
        with open(heartbeat_file, "w", encoding="utf-8") as f:
            f.write(str(time.time()))
    except OSError:
        pass


def _heartbeat_is_fresh(heartbeat_file):
    try:
        return (time.time() - os.path.getmtime(heartbeat_file)) < HEARTBEAT_STALE_SECONDS
    except OSError:
        return False


def _clear_state_files(pid_file, heartbeat_file):
    for path in (pid_file, heartbeat_file):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def _kill_pid(pid):
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


def _kill_existing_listener(pid_file, heartbeat_file):
    """Last-writer-wins: a zombie listener from a dead session would race us
    to archive mail before the agent sees it. Only kill when the heartbeat is
    fresh — a stale one means the PID may have been recycled by an unrelated
    process (Windows PID reuse)."""
    try:
        with open(pid_file, encoding="utf-8") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        _clear_state_files(pid_file, heartbeat_file)
        return
    if pid != os.getpid() and _heartbeat_is_fresh(heartbeat_file):
        _kill_pid(pid)
    _clear_state_files(pid_file, heartbeat_file)


def _letters(inbox):
    if not os.path.isdir(inbox):
        return []
    return sorted(
        f
        for f in os.listdir(inbox)
        if f.endswith(EXTENSIONS) and os.path.isfile(os.path.join(inbox, f))
    )


def _archive(inbox, read_dir, basename):
    src = os.path.join(inbox, basename)
    dst = os.path.join(read_dir, basename)
    for _ in range(50):  # 5s of retries rides out Windows file-lock races
        try:
            shutil.move(src, dst)
            return
        except PermissionError:
            time.sleep(0.1)
        except FileNotFoundError:
            return  # another listener already filed it
    print(f"WARNING: could not archive {basename}", file=sys.stderr)


def _deliver(name, inbox, read_dir, names):
    for basename in names:
        path = os.path.join(inbox, basename)
        try:
            with open(path, encoding="utf-8") as f:
                body = f.read()
        except OSError:
            body = ""
        _archive(inbox, read_dir, basename)
        print(f"=== {name} mail: {basename} ===")
        print(body)
    print(f"MAIL:{len(names)}:{','.join(names)}")
    print("---")
    print("Listener exited on mail. Re-arm now:")
    print(f"  python .agent-runtime/listen_for_mail.py --name {name} 86000")


def main():
    parser = argparse.ArgumentParser(description="Inbox listener for agent mail.")
    parser.add_argument(
        "--name",
        required=True,
        choices=("kimi", "claude"),
        help="Whose mailbox to watch.",
    )
    parser.add_argument(
        "timeout",
        type=int,
        help="Seconds to wait before timeout.",
    )
    args = parser.parse_args()

    inbox, read_dir, pid_file, heartbeat_file = _derive_paths(args.name)

    os.makedirs(inbox, exist_ok=True)
    os.makedirs(read_dir, exist_ok=True)
    _kill_existing_listener(pid_file, heartbeat_file)

    try:
        with open(pid_file, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except OSError as e:
        print(f"WARNING: pid write failed: {e}", file=sys.stderr)
    _touch_heartbeat(heartbeat_file)
    start = time.time()

    try:
        pending = _letters(inbox)  # drain mail that predates this arm
        if pending:
            _deliver(args.name, inbox, read_dir, pending)
            return
        while True:
            time.sleep(POLL_SECONDS)
            _touch_heartbeat(heartbeat_file)
            if not os.path.isdir(inbox):
                print("ERROR: inbox directory was deleted", file=sys.stderr)
                sys.exit(1)
            pending = _letters(inbox)
            if pending:
                _deliver(args.name, inbox, read_dir, pending)
                return
            if time.time() - start > args.timeout:
                print(f"TIMEOUT: no mail after {args.timeout}s. Re-arm now:")
                print(f"  python .agent-runtime/listen_for_mail.py --name {args.name} {args.timeout}")
                return
    finally:
        _clear_state_files(pid_file, heartbeat_file)


if __name__ == "__main__":
    main()
