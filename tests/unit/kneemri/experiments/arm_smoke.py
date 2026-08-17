"""What the smoke sweep runs, and what it calls a failure."""

import json
from pathlib import Path

from kneemri.experiments.arm_smoke import ArmSmoke


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
