from pathlib import Path

from kneemri.experiments.queue_check import QueuedArms, Watch

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


def test_listing():
    """An empty table reads as 'nothing is running' and would strand every queued arm, so it is checked."""
    listing = QueuedArms.listing()
    assert "PID" in listing.splitlines()[0], "the header `ps -ef` prints, so the table was actually read"
    assert len(listing.splitlines()) > 1, "a machine running this test has at least one process"


def test_lines_empty(tmp_path: Path):
    assert QueuedArms(tmp_path, LIVE).lines()[-1] == "OK: nothing is queued"
