from pathlib import Path

import pytest

from kneemri.experiments.queue_check import Probe, QueuedArms, Watch, main

MARKER = "ARM DONE aug_long"
ARMS = "experiments/arms/v3base_band.sh experiments/arms/aprior_dense.sh"
LIVE = f"User 107665 107663 ? bash experiments/after.sh {MARKER} /d/x.log {ARMS}\n"


def _log(directory: Path, name: str, marker: str, arms: str, started: int = 0) -> Path:
    path = directory / name
    body = [f"waiting for '{marker}' in /d/watch.log before running: {arms}"]
    body += [f"=== {arm} ===" for arm in arms.split()[:started]]
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def test_of(tmp_path: Path):
    watch = Watch.of(_log(tmp_path, "after_chain.log", MARKER, ARMS, started=1))
    assert watch is not None
    assert watch.marker == MARKER, "the marker is quoted in the header and is what liveness is crossed on"
    assert watch.arms == ("experiments/arms/v3base_band.sh", "experiments/arms/aprior_dense.sh")
    assert watch.started == 1


def test_of_not_a_watcher(tmp_path: Path):
    (tmp_path / "after_gates.log").write_text("nox > Session lint was successful", encoding="utf-8")
    assert Watch.of(tmp_path / "after_gates.log") is None, "a gate log next door is not a broken queue"


def test_waiting(tmp_path: Path):
    watch = Watch.of(_log(tmp_path, "after_chain.log", MARKER, ARMS, started=1))
    assert watch is not None
    assert watch.waiting() == ("experiments/arms/aprior_dense.sh",), "the announced arm is not still owed"


def test_watches(tmp_path: Path):
    _log(tmp_path, "after_chain.log", MARKER, ARMS)
    (tmp_path / "push.log").write_text("144da18..9c33422  main -> main", encoding="utf-8")
    assert [watch.log for watch in QueuedArms(tmp_path, LIVE).watches()] == ["after_chain.log"]


def test_alive(tmp_path: Path):
    _log(tmp_path, "after_chain.log", MARKER, ARMS)
    watch = QueuedArms(tmp_path, LIVE).watches()[0]
    assert QueuedArms(tmp_path, LIVE).alive(watch), "a marker in the listing's argv means a live gate"
    assert not QueuedArms(tmp_path, "").alive(watch), "an empty listing leaves nothing gating it"


def test_stranded(tmp_path: Path):
    _log(tmp_path, "after_chain.log", MARKER, ARMS)
    assert QueuedArms(tmp_path, LIVE).stranded() == [], "a live watcher owes nothing to anyone else"
    assert QueuedArms(tmp_path, "").stranded() == ARMS.split(), "no process, so both arms are owed by nobody"


def test_stranded_covered_by_a_relaunch(tmp_path: Path):
    """The incident's own recovery: three dead logs whose arms a new live chain had taken over."""
    _log(tmp_path, "after_v3band.log", "ARM DONE dead", "experiments/arms/v3base_band.sh")
    _log(tmp_path, "after_chain.log", MARKER, ARMS)
    assert QueuedArms(tmp_path, LIVE).stranded() == [], "an arm a live chain owes is not stranded elsewhere"


def test_stranded_already_on_the_card(tmp_path: Path):
    """This check's first live run: an arm training right now, still claimed by a dead log."""
    _log(tmp_path, "after_chain.log", MARKER, ARMS, started=2)
    _log(tmp_path, "after_aprior.log", "ARM DONE dead", "experiments/arms/aprior_dense.sh")
    assert QueuedArms(tmp_path, LIVE).stranded() == [], "an announced arm has already run, not gone missing"


def test_lines(tmp_path: Path):
    _log(tmp_path, "after_chain.log", MARKER, ARMS)
    assert QueuedArms(tmp_path, LIVE).lines()[-1].startswith("OK:")
    stranded = QueuedArms(tmp_path, "").lines()[-1]
    assert stranded.startswith("STRANDED:") and "v3base_band" in stranded, "the verdict names the arms"


def test_read():
    """A probe whose executable is not on THIS interpreter's PATH answers empty rather than raising.

    MEASURED 2026-08-20: `shutil.which("ps")` returns None from the Python process this check runs in
    — the MSYS `ps` lives in git-bash's `/usr/bin`, off the Windows PATH — while `ps -ef` works fine in
    the shell that launches it. That gap is what made the check silently pass on the only machine it
    runs on, so the absent case is a test rather than an assumption.
    """
    assert Probe("no-such-executable-anywhere", ()).read() == ""


def test_listing():
    """An empty table reads as 'nothing is running' and would strand every queued arm, so it is checked.

    Asserted on CONTENT, not on a header: the two probes print different tables, and the crossing only
    ever matches command lines, so what must hold is that some process this test knows is running is
    named in there.

    SKIPS RATHER THAN FAILS WHEN NO PROBE SPAWNS, and that is not a silent pass. On 2026-08-20 a
    pre-push hook running beside a training arm, a CPU witness over nineteen members and a full gate
    session failed fourteen tests with `exit status 3221225794` — Windows `STATUS_DLL_INIT_FAILED`,
    the OS refusing to start any new process. The same tree passed serially minutes later. That is a
    fact about the machine's load, not about this code, and a check that cannot spawn `ps` has not
    measured that `ps` is broken. The condition this test guards is enforced where it matters instead:
    `queue_check.main` now EXITS NON-ZERO on an unreadable table, so a genuinely broken probe fails the
    check every time it is run for real, while a momentarily overloaded machine no longer blocks a push.
    """
    listing = QueuedArms.listing()
    if not listing:
        pytest.skip("no probe could be spawned on this machine right now — see test_listing_no_probe_reads_empty")
    assert len(listing.splitlines()) > 1, "a machine running this test has at least one process"
    assert "python" in listing.lower(), "the interpreter running this test is a process and must be listed"


def test_listing_no_probe_reads_empty():
    """No probe means no answer — and the caller must be told, not handed a table with nothing in it."""
    assert QueuedArms.listing((Probe("no-such-executable-anywhere", ()),)) == ""


def test_lines_empty(tmp_path: Path):
    assert QueuedArms(tmp_path, LIVE).lines()[-1] == "OK: nothing is queued"


def test_main_exits_when_the_table_cannot_be_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """An unreadable process table must not leave the status at 0 — that reads as a pass in a log tail."""
    monkeypatch.setattr(QueuedArms, "listing", staticmethod(lambda *_: ""))
    monkeypatch.setattr("sys.argv", ["queue_check", "--directory", str(tmp_path)])
    with pytest.raises(SystemExit) as exit_code:
        main()
    assert exit_code.value.code == 1
