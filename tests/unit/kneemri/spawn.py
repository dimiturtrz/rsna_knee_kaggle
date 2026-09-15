"""The spawn retry, over the four things a child process can do to us."""

import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from kneemri import spawn
from kneemri.spawn import Spawn

Run = Callable[..., subprocess.CompletedProcess[str]]


def _fake_runs(*codes: int) -> tuple[Run, list[list[str]]]:
    """A `subprocess.run` that hands back the given exit codes in order, and records what it was asked."""
    calls: list[list[str]] = []
    remaining = list(codes)

    def run(command: list[str], **_options: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        return subprocess.CompletedProcess(command, remaining.pop(0), stdout="out", stderr="err")

    return run, calls


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The backoff is real seconds and a unit test does not spend them."""
    monkeypatch.setattr(spawn.time, "sleep", lambda _s: None)
    yield


def test_ran(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean child is returned as it is, on one attempt."""
    run, calls = _fake_runs(0)
    monkeypatch.setattr(spawn.subprocess, "run", run)
    assert Spawn().ran(["git", "status"]).returncode == 0
    assert len(calls) == 1


def test_ran_retries_a_child_that_never_started(monkeypatch: pytest.MonkeyPatch) -> None:
    run, calls = _fake_runs(spawn.STATUS_DLL_INIT_FAILED, 0)
    monkeypatch.setattr(spawn.subprocess, "run", run)
    assert Spawn().ran(["git", "init"]).returncode == 0
    assert len(calls) == 2


def test_ran_gives_up_when_the_box_stays_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 3
    run, calls = _fake_runs(*([spawn.STATUS_DLL_INIT_FAILED] * attempts))
    monkeypatch.setattr(spawn.subprocess, "run", run)
    with pytest.raises(OSError, match="too loaded"):
        Spawn(retries=attempts).ran(["git", "init"])
    assert len(calls) == attempts


def test_ran_never_retries_a_real_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: a check that failed on its own terms is reported, not run until it agrees."""
    run, calls = _fake_runs(1)
    monkeypatch.setattr(spawn.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        Spawn().ran(["git", "rev-parse"], check=True)
    assert len(calls) == 1


def test_attempted(monkeypatch: pytest.MonkeyPatch) -> None:
    """`check` is withheld from the child, or a raise would hide the code the retry dispatches on."""
    seen: dict[str, object] = {}

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        seen.update(options)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    monkeypatch.setattr(spawn.subprocess, "run", run)
    assert Spawn().attempted(["git", "status"], check=True).returncode == 1
    assert seen["check"] is False


def test_checked() -> None:
    """A nonzero child raises only when the caller asked for `check`."""
    done = subprocess.CompletedProcess(["git"], 1, stdout="", stderr="")
    assert Spawn().checked(done, ["git"]).returncode == 1
    with pytest.raises(subprocess.CalledProcessError):
        Spawn().checked(done, ["git"], check=True)


def _alive(pid: int) -> bool:
    """Whether a pid still runs, asked without touching it — `os.kill(pid, 0)` TERMINATES on Windows."""
    if os.name == "nt":
        listed = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, check=False)  # noqa: S603, S607
        return str(pid).encode() in listed.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_bounded() -> None:
    """A child that finishes in time is returned whole, output included."""
    done = Spawn().bounded([sys.executable, "-c", "print('hi')"], 60)
    assert done is not None
    assert (done.returncode, done.stdout.strip()) == (0, "hi")


def test_bounded_kills_the_grandchild_on_overrun(tmp_path: Path) -> None:
    """Overrun answers None AND leaves no descendant computing — the orphaned GPU pretrain of knee-u2aet."""
    marker = tmp_path / "grandchild.pid"
    body = (
        "import pathlib, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import threading; threading.Event().wait()'])\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(child.pid))\n"
        "child.wait()\n"
    )
    assert Spawn().bounded([sys.executable, "-c", body], 5) is None
    assert not _alive(int(marker.read_text())), "the tree has to go, not just its root"


def test_killed() -> None:
    """A killed process is gone."""
    child = subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE, start_new_session=True)
    Spawn.killed(child.pid)
    assert child.wait(timeout=30) != 0
