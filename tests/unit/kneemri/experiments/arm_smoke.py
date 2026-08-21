"""What the smoke sweep runs, and what it calls a failure."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from kneemri.experiments.arm_smoke import SMOKE_TIMEOUT_SECONDS, ArmSmoke

LAST_SWEEP = 1_700_000_000
"""A stamp from before this run, so "the smoke wrote it" is decided by the clock rather than by timing."""


def test_command(tmp_path: Path):
    """The smoke's three overrides come LAST, which is how they beat the arm's own values."""
    argv = ArmSmoke(studies=25).command("--fold 0 --epochs 12 --out real.pt", tmp_path / "smoke.pt")

    assert argv[1:3] == ["-m", "kneemri.training.train"], "the trainer itself, not a stub of it"
    assert argv[-8:] == [
        *("--cache-studies", "25"),
        *("--studies", "25"),
        *("--epochs", "1"),
        *("--out", str(tmp_path / "smoke.pt")),
    ], "the truncated CACHE as well as the subsample, or an untried reading has nothing to read"
    assert argv.index("--epochs") < len(argv) - 8, "the arm's own --epochs stays, and argparse keeps the last"


def _record(out: Path, name: str, loss: float, auc: float, fold: str = "0") -> None:
    """The two lines a real run leaves beside its checkpoint: the recipe, then the epoch."""
    body = json.dumps({"recipe": {"fold": fold}}) + "\n" + json.dumps({"epoch": 0, "loss": loss, "auc": auc}) + "\n"
    (out / f"{name}.metrics.jsonl").write_text(body, encoding="utf-8")


def test_failed(tmp_path: Path):
    """A non-zero exit is the arm's failure and carries its last complaint; a zero exit must still train."""
    said = {
        "good": (0, ""),
        "diverged": (0, ""),
        "oom": (1, "torch.OutOfMemoryError: CUDA out of memory."),
        "mute": (2, ""),
    }

    def runner(argv: list[str]) -> tuple[int, str]:
        name = Path(argv[-1]).stem
        _record(tmp_path, name, loss=float("nan") if name == "diverged" else 0.62, auc=0.55)
        (tmp_path / f"{name}.partial").write_bytes(b"weights")
        (tmp_path / f"{name}.holdout.npz").write_bytes(b"predictions")
        return said[name]

    (tmp_path / "good.sh").write_text("run_train good\n", encoding="utf-8")
    broken = ArmSmoke(runner=runner).failed(dict.fromkeys(said, "--fold 0"), tmp_path)

    assert "good" not in broken, "an arm that survives an epoch is not reported"
    assert broken["diverged"] == "epoch 0 recorded a non-finite loss", "exit 0 with a NaN loss is a broken recipe"
    assert broken["oom"] == "torch.OutOfMemoryError: CUDA out of memory.", "the last thing it said"
    assert broken["mute"] == "exit 2 with nothing on stderr", "a silent failure is still a failure"
    assert not list(tmp_path.glob("*.partial")), "the periodic save's weights are cleaned up, not only the .pt"
    assert not list(tmp_path.glob("*.npz")), "every sibling the arm wrote goes, not the suffixes someone listed"
    assert (tmp_path / "good.sh").exists(), "a file that was there BEFORE the arm ran is not the arm's output"


def test_failed_reports_the_arm_that_hung(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A timeout is that arm's result, and the sweep goes on to the next one.

    `subprocess.run(timeout=…)` RAISES, and nothing caught it: the hang the timeout exists to bound
    took the whole sweep down instead of naming the arm that hung, and left that arm's artifacts
    behind because the cleanup sits past the call.
    """

    def hangs(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if Path(argv[-1]).stem == "hangs":
            (tmp_path / "hangs.partial").write_bytes(b"what it wrote before it stopped moving")
            raise subprocess.TimeoutExpired(argv, SMOKE_TIMEOUT_SECONDS)
        _record(tmp_path, "after", loss=0.62, auc=0.55)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("kneemri.spawn.subprocess.run", hangs)
    stale = tmp_path / "hangs.partial"
    stale.write_bytes(b"the previous sweep's weights")
    os.utime(stale, (LAST_SWEEP, LAST_SWEEP))
    broken = ArmSmoke().failed({"hangs": "--fold 0", "after": "--fold 0"}, tmp_path)

    assert str(SMOKE_TIMEOUT_SECONDS) in broken["hangs"], "the arm that hung is named, with the bound it broke"
    assert "after" not in broken, "and the arms queued behind it still run"
    assert not list(tmp_path.glob("*.partial")), "a hung arm's artifacts are cleaned like any other's"


def test_failed_cleans_a_leftover_it_rewrote(tmp_path: Path):
    """A second sweep into the same directory must clean the artifacts it OVERWROTE, not only new names.

    The cleanup used to skip every path that was there beforehand — which protects the arm file, and
    protected the previous sweep's `.partial` too, since the trainer rewrites it under the same name.
    Measured 2026-08-17: 3.5 GB of ninety-megabyte checkpoints sitting in the directory whose whole
    point is to hold none. The stamp is what tells the two apart.
    """
    stale = tmp_path / "again.partial"
    stale.write_bytes(b"the previous sweep's weights")
    os.utime(stale, (LAST_SWEEP, LAST_SWEEP))
    (tmp_path / "again.sh").write_text("run_train again\n", encoding="utf-8")
    os.utime(tmp_path / "again.sh", (LAST_SWEEP, LAST_SWEEP))

    def runner(argv: list[str]) -> tuple[int, str]:
        name = Path(argv[-1]).stem
        _record(tmp_path, name, loss=0.62, auc=0.55)
        stale.write_bytes(b"this sweep's weights")
        return 0, ""

    ArmSmoke(runner=runner).failed({"again": "--fold 0"}, tmp_path)

    assert not stale.exists(), "a leftover the smoke rewrote is this run's output, whatever its name was before"
    assert (tmp_path / "again.sh").exists(), "and the arm file, which nothing touched, stays"


def test_learned(tmp_path: Path):
    """The epoch's own aggregates decide, and a per-label NaN at sixty studies is not a fault."""
    smoke = ArmSmoke()
    _record(tmp_path, "fine", loss=0.62, auc=0.55)
    (tmp_path / "fine.metrics.jsonl").write_text(
        json.dumps({"epoch": 0, "loss": 0.62, "auc": 0.55, "per_label": {"Fracture": float("nan")}}) + "\n",
        encoding="utf-8",
    )
    _record(tmp_path, "blown", loss=float("nan"), auc=float("nan"))
    (tmp_path / "started.metrics.jsonl").write_text(json.dumps({"recipe": {"fold": "0"}}) + "\n", encoding="utf-8")

    assert smoke.learned(tmp_path / "fine.pt") == "", "a rare finding with one class in the holdout is arithmetic"
    assert smoke.learned(tmp_path / "blown.pt") == "epoch 0 recorded a non-finite loss and auc", "both named"
    _record(tmp_path, "deliverable", loss=1.34, auc=float("nan"), fold="-1")

    assert smoke.learned(tmp_path / "deliverable.pt") == "", "--fold -1 keeps no holdout, so its NaN AUC is the design"
    assert smoke.learned(tmp_path / "started.pt") == "exited 0 without finishing an epoch"
    assert smoke.learned(tmp_path / "absent.pt") == "exited 0 but recorded no metrics at all"


def test_reported(tmp_path: Path):
    """A refused arm is never run, and the directory the checkpoints want is made rather than assumed."""
    ran: list[str] = []

    def runner(argv: list[str]) -> tuple[int, str]:
        checkpoint = Path(argv[-1])
        ran.append(checkpoint.stem)
        _record(checkpoint.parent, checkpoint.stem, loss=0.62, auc=0.55)
        return 0, ""

    out = tmp_path / "not_yet_there"
    broken = ArmSmoke(runner=runner).reported({"fine": "--fold 0", "typo": "--fld 0"}, {"typo": "no such flag"}, out)

    assert ran == ["fine"], "an arm the parser already refused is not worth the card"
    assert not broken, "nothing failed, so nothing is reported"
    assert out.is_dir(), "the smoke makes its own output directory"
