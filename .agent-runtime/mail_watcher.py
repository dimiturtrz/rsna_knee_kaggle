#!/usr/bin/env python3
"""Single-process threaded mail watcher — exits on mail, re-arms on timeout.

Watches Kimi's mailbox at .agent-runtime/inbox/ and Claude's at .claude/inbox/.
When either mailbox receives a letter, this process exits so the harness raises a
background-task completion notification. On listener timeout or error the watcher
re-arms automatically.

Arm as ONE harness background task with disable_timeout=true. Re-arm immediately
after each notification.
"""

import argparse
import os
import shutil
import sys
import threading
import time

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

POLL_SECONDS = 2
HEARTBEAT_STALE_SECONDS = 30
EXTENSIONS = (".txt", ".md")


class _Done(Exception):
    pass


def _log(path, line):
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{stamp} {line}\n")
    except OSError:
        pass


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
    for _ in range(50):
        try:
            shutil.move(src, dst)
            return
        except PermissionError:
            time.sleep(0.1)
        except FileNotFoundError:
            return
    print(f"WARNING: could not archive {basename}", file=sys.stderr)


def _derive_paths(name):
    if name == "claude":
        agent_dir = os.path.join(REPO_ROOT, ".claude")
    else:
        agent_dir = SCRIPT_DIR
    inbox_dir = os.path.join(agent_dir, "inbox")
    read_dir = os.path.join(inbox_dir, "read")
    return inbox_dir, read_dir


def _watch_one(name, timeout, log_path, done_event):
    inbox_dir, read_dir = _derive_paths(name)
    os.makedirs(inbox_dir, exist_ok=True)
    os.makedirs(read_dir, exist_ok=True)

    while not done_event.is_set():
        _log(log_path, f"[{name}] starting listener")
        start = time.time()
        try:
            pending = _letters(inbox_dir)
            if pending:
                for basename in pending:
                    path = os.path.join(inbox_dir, basename)
                    try:
                        with open(path, encoding="utf-8") as f:
                            body = f.read()
                    except OSError:
                        body = ""
                    _archive(inbox_dir, read_dir, basename)
                    print(f"=== {name} mail: {basename} ===")
                    print(body)
                print(f"MAIL:{len(pending)}:{','.join(pending)}")
                _log(log_path, f"[{name}] delivered {len(pending)} letter(s), watcher exiting")
                done_event.set()
                return

            while not done_event.is_set():
                time.sleep(POLL_SECONDS)
                if not os.path.isdir(inbox_dir):
                    _log(log_path, f"[{name}] inbox deleted, exiting")
                    return
                pending = _letters(inbox_dir)
                if pending:
                    for basename in pending:
                        path = os.path.join(inbox_dir, basename)
                        try:
                            with open(path, encoding="utf-8") as f:
                                body = f.read()
                        except OSError:
                            body = ""
                        _archive(inbox_dir, read_dir, basename)
                        print(f"=== {name} mail: {basename} ===")
                        print(body)
                    print(f"MAIL:{len(pending)}:{','.join(pending)}")
                    _log(log_path, f"[{name}] delivered {len(pending)} letter(s), watcher exiting")
                    done_event.set()
                    return
                if time.time() - start > timeout:
                    _log(log_path, f"[{name}] listener timed out, re-arming")
                    break
        except Exception as exc:  # noqa: BLE001
            _log(log_path, f"[{name}] listener error: {exc}, re-arming")
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description="Single-process threaded mail watcher.")
    parser.add_argument(
        "--name",
        default="both",
        choices=("kimi", "claude", "both"),
        help="Which mailbox to watch.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=86000,
        help="Listener timeout in seconds.",
    )
    args = parser.parse_args()

    log_path = os.path.join(SCRIPT_DIR, "mail_alerts.kimi.log")
    claude_log_path = os.path.join(REPO_ROOT, ".claude", "mail_alerts.claude.log")

    names = []
    if args.name in ("kimi", "both"):
        names.append(("kimi", log_path))
    if args.name in ("claude", "both"):
        names.append(("claude", claude_log_path))

    done_event = threading.Event()
    threads = []
    for name, log in names:
        t = threading.Thread(
            target=_watch_one,
            args=(name, args.timeout, log, done_event),
            daemon=True,
        )
        t.start()
        threads.append(t)

    try:
        while not done_event.is_set():
            time.sleep(0.5)
            if all(not t.is_alive() for t in threads):
                break
    except KeyboardInterrupt:
        done_event.set()

    for t in threads:
        t.join(timeout=5)

    if done_event.is_set():
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
