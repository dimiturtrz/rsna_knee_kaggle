"""What the smoke sweep runs, and what it calls a failure."""

from pathlib import Path

from kneemri.experiments.arm_smoke import ArmSmoke


def test_command(tmp_path: Path):
    """The smoke's three overrides come LAST, which is how they beat the arm's own values."""
    argv = ArmSmoke(studies=25).command("--fold 0 --epochs 12 --out real.pt", tmp_path / "smoke.pt")

    assert argv[1:3] == ["-m", "kneemri.training.train"], "the trainer itself, not a stub of it"
    assert argv[-6:] == ["--studies", "25", "--epochs", "1", "--out", str(tmp_path / "smoke.pt")]
    assert argv.index("--epochs") < len(argv) - 6, "the arm's own --epochs stays, and argparse keeps the last"


def test_failed(tmp_path: Path):
    """A non-zero exit is the arm's failure and carries its last complaint; a zero exit says nothing."""
    said = {"good": (0, ""), "oom": (1, "torch.OutOfMemoryError: CUDA out of memory."), "mute": (2, "")}
    smoke = ArmSmoke(runner=lambda argv: said[Path(argv[-1]).stem])

    broken = smoke.failed(dict.fromkeys(said, "--fold 0"), tmp_path)

    assert "good" not in broken, "an arm that survives an epoch is not reported"
    assert broken["oom"] == "torch.OutOfMemoryError: CUDA out of memory.", "the last thing it said"
    assert broken["mute"] == "exit 2 with nothing on stderr", "a silent failure is still a failure"


def test_reported(tmp_path: Path):
    """A refused arm is never run, and the directory the checkpoints want is made rather than assumed."""
    ran: list[str] = []

    def runner(argv: list[str]) -> tuple[int, str]:
        ran.append(Path(argv[-1]).stem)
        return 0, ""

    out = tmp_path / "not_yet_there"
    broken = ArmSmoke(runner=runner).reported({"fine": "--fold 0", "typo": "--fld 0"}, {"typo": "no such flag"}, out)

    assert ran == ["fine"], "an arm the parser already refused is not worth the card"
    assert not broken, "nothing failed, so nothing is reported"
    assert out.is_dir(), "the smoke makes its own output directory"
