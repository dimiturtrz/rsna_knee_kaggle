from pathlib import Path

import pytest

from kneemri.experiments.queue_check import QueuedArms, Watch, main
from kneemri.process_table import ProcessTable

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


def test_alive_is_not_vouched_for_by_the_chain_that_replaced_it(tmp_path: Path):
    """A superseded watcher's argv is a SUFFIX of the chain that took over — the 2026-08-21 hole.

    The chain owes both arms; the dead log owes only the second. Matching the marker, or matching the
    arms anywhere in the line, calls the dead watcher live and the queue healthy.
    """
    _log(tmp_path, "after_dead.log", MARKER, "experiments/arms/aprior_dense.sh")
    dead = QueuedArms(tmp_path, LIVE).watches()[0]
    assert not QueuedArms(tmp_path, LIVE).alive(dead), "a longer chain does not vouch for a killed watcher"


def test_watchers(tmp_path: Path):
    """Only `after.sh` lines gate; the arm's own process carries the same path and is not a second gate."""
    running = LIVE + "User 1 1 ? bash experiments/arms/aprior_dense.sh\n"
    assert QueuedArms(tmp_path, running).watchers() == [LIVE.rstrip()]


def test_watchers_drops_the_launching_shell(tmp_path: Path):
    """The harness shell's own `-c` string embeds the nohup line, so one queue shows up as two.

    MEASURED 2026-08-21: this is what a naive count reported as DOUBLE-QUEUED on a healthy queue —
    a false blocking verdict is as expensive as the miss it replaced.
    """
    wrapper = f"User 107663 1 ? bash -c cd repo && nohup bash experiments/after.sh {MARKER} /d/x.log {ARMS}\n"
    assert QueuedArms(tmp_path, wrapper + LIVE).watchers() == [LIVE.rstrip()]


def test_doubled(tmp_path: Path):
    """Two live watchers each owing one arm — 2.5 h of slot, and two of that geometry overflow the card."""
    _log(tmp_path, "after_chain.log", MARKER, ARMS)
    assert QueuedArms(tmp_path, LIVE).doubled() == [], "one watcher per arm is the healthy queue"
    assert QueuedArms(tmp_path, LIVE + LIVE).doubled() == ARMS.split(), "both arms would start twice"


def test_faults(tmp_path: Path):
    _log(tmp_path, "after_chain.log", MARKER, ARMS)
    assert QueuedArms(tmp_path, LIVE).faults() == []
    assert QueuedArms(tmp_path, LIVE + LIVE).faults()[0].startswith("DOUBLE-QUEUED:")
    assert QueuedArms(tmp_path, "").faults()[0].startswith("STRANDED:")


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


def test_lines_empty(tmp_path: Path):
    assert QueuedArms(tmp_path, LIVE).lines()[-1] == "OK: nothing is queued"


def test_main_exits_when_the_table_cannot_be_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """An unreadable process table must not leave the status at 0 — that reads as a pass in a log tail."""
    monkeypatch.setattr(ProcessTable, "read", classmethod(lambda *_: ProcessTable("")))
    monkeypatch.setattr("sys.argv", ["queue_check", "--directory", str(tmp_path)])
    with pytest.raises(SystemExit) as exit_code:
        main()
    assert exit_code.value.code == 1


def test_main_exits_on_a_fault(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A stranded queue must not leave the status at 0 either — same trap, and it costs a card."""
    _log(tmp_path, "after_chain.log", MARKER, ARMS)
    monkeypatch.setattr(ProcessTable, "read", classmethod(lambda *_: ProcessTable("bash nothing\n")))
    monkeypatch.setattr("sys.argv", ["queue_check", "--directory", str(tmp_path)])
    with pytest.raises(SystemExit) as exit_code:
        main()
    assert exit_code.value.code == 2
