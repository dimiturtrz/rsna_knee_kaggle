#!/usr/bin/env python3
"""Inbox listener — arm as a harness background task, exits on mail.

Watches .agent-runtime/inbox/ for .txt/.md letters. On arrival: prints each
letter's body, archives it to inbox/read/, and exits so the harness delivers
the output as a background-task completion notification. Re-arm FIRST on every
notification, then read, then work.

Stdlib-only polling: filesystem-event listeners (watchdog) are a dependency
this repo does not otherwise need, and events were unreliable enough in the
fpga project that its listener drained on a 5s poll regardless.

Usage: python .agent-runtime/listen_for_mail.py [timeout_seconds]
Arm with a harness timeout slightly above the internal one.
"""

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
DEFAULT_TIMEOUT_SECONDS = 82800  # arm the background task with 86400

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INBOX = os.path.join(SCRIPT_DIR, "inbox")
READ_DIR = os.path.join(INBOX, "read")
PID_FILE = os.path.join(INBOX, ".listener_pid")
HEARTBEAT_FILE = os.path.join(INBOX, ".listener_heartbeat")

EXTENSIONS = (".txt", ".md")


def _touch_heartbeat():
    try:
        with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
            f.write(str(time.time()))
    except OSError:
        pass


def _heartbeat_is_fresh():
    try:
        return (time.time() - os.path.getmtime(HEARTBEAT_FILE)) < HEARTBEAT_STALE_SECONDS
    except OSError:
        return False


def _clear_state_files():
    for path in (PID_FILE, HEARTBEAT_FILE):
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


def _kill_existing_listener():
    """Last-writer-wins: a zombie listener from a dead session would race us
    to archive mail before the agent sees it. Only kill when the heartbeat is
    fresh — a stale one means the PID may have been recycled by an unrelated
    process (Windows PID reuse)."""
    try:
        with open(PID_FILE, encoding="utf-8") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        _clear_state_files()
        return
    if pid != os.getpid() and _heartbeat_is_fresh():
        _kill_pid(pid)
    _clear_state_files()


def _letters():
    if not os.path.isdir(INBOX):
        return []
    return sorted(f for f in os.listdir(INBOX) if f.endswith(EXTENSIONS) and os.path.isfile(os.path.join(INBOX, f)))


def _archive(basename):
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


def _deliver(names):
    for basename in names:
        path = os.path.join(INBOX, basename)
        try:
            with open(path, encoding="utf-8") as f:
                body = f.read()
        except OSError:
            body = ""
        _archive(basename)
        print(f"=== inbox/{basename} ===")
        print(body)
    print(f"MAIL:{len(names)}:{','.join(names)}")
    print("---")
    print("Listener exited on mail. Re-arm now:")
    print("  python .agent-runtime/listen_for_mail.py")


def main():
    timeout = DEFAULT_TIMEOUT_SECONDS
    if len(sys.argv) >= 2:
        try:
            timeout = int(sys.argv[1])
        except ValueError:
            print(f"invalid timeout: {sys.argv[1]}", file=sys.stderr)
            sys.exit(2)

    os.makedirs(READ_DIR, exist_ok=True)
    _kill_existing_listener()

    try:
        with open(PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except OSError as e:
        print(f"WARNING: pid write failed: {e}", file=sys.stderr)
    _touch_heartbeat()
    start = time.time()

    try:
        pending = _letters()  # drain mail that predates this arm
        if pending:
            _deliver(pending)
            return
        while True:
            time.sleep(POLL_SECONDS)
            _touch_heartbeat()
            if not os.path.isdir(INBOX):
                print("ERROR: inbox directory was deleted", file=sys.stderr)
                sys.exit(1)
            pending = _letters()
            if pending:
                _deliver(pending)
                return
            if time.time() - start > timeout:
                print(f"TIMEOUT: no mail after {timeout}s. Re-arm now:")
                print(f"  python .agent-runtime/listen_for_mail.py {timeout}")
                return
    finally:
        _clear_state_files()


if __name__ == "__main__":
    main()
