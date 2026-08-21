"""The spawn retry, over the four things a child process can do to us."""

import subprocess
from collections.abc import Callable, Iterator

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
