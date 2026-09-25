"""Whether a deployed member differs from the twin that was validated on fold alone."""

from pathlib import Path

import torch

from kneemri.experiments.twin_check import TwinDrift

VALIDATED: dict[str, object] = {"fold": 0, "ema": False, "out": "twin.pt", "epochs": 24}
DEPLOYED: dict[str, object] = {"fold": -1, "ema": True, "out": "ff.pt", "epochs": 24}


def stamp(path: Path, options: dict[str, object], config: dict[str, object] | None = None) -> Path:
    """A checkpoint carrying only what `StampedRun` reads off it."""
    torch.save({"options": options, "config": config or {}, "epoch": 1}, path)
    return path


def pair(models: Path, options: dict[str, object]) -> TwinDrift:
    """One deployed/validated pair, the deployed half carrying `options`."""
    stamp(models / "arm_ff.pt", options)
    stamp(models / "arm_s2026.pt", VALIDATED)
    return TwinDrift(models)


def test_pairs(tmp_path: Path) -> None:
    """A deployed checkpoint is paired with its twin by name, and an unpaired one is not compared."""
    stamp(tmp_path / "arm_ff.pt", DEPLOYED)
    stamp(tmp_path / "arm_s2026.pt", VALIDATED)
    stamp(tmp_path / "lonely_ff.pt", DEPLOYED)
    assert [deployed.name for deployed, _ in TwinDrift(tmp_path).pairs()] == ["arm_ff.pt"]


def test_drift(tmp_path: Path) -> None:
    """The allowance passes, a real second variable does not, and a flag one side predates is not one."""
    assert pair(tmp_path, DEPLOYED).drift(tmp_path / "arm_ff.pt", tmp_path / "arm_s2026.pt") == []

    moved = pair(tmp_path, {**DEPLOYED, "epochs": 12})
    assert moved.drift(tmp_path / "arm_ff.pt", tmp_path / "arm_s2026.pt") == ["epochs='24'->'12'"]

    # AN AXIS AT ITS DEFAULT ON ONE SIDE AND ABSENT ON THE OTHER IS ONE MODEL: a run predating the flag
    # ran at that default. Only a value BOTH sides state, differently, is a second variable.
    era = pair(tmp_path, DEPLOYED)
    stamp(tmp_path / "arm_s2026.pt", {**VALIDATED, "layer_decay": 1.0})
    assert era.drift(tmp_path / "arm_ff.pt", tmp_path / "arm_s2026.pt") == []


def test_faults(tmp_path: Path) -> None:
    """A pair differing on the allowance alone is silent; one carrying another axis names it."""
    assert pair(tmp_path, DEPLOYED).faults() == []

    stamp(tmp_path / "arm_ff.pt", {**DEPLOYED, "slices": 24})
    stamp(tmp_path / "arm_s2026.pt", {**VALIDATED, "slices": 20})
    assert "slices" in TwinDrift(tmp_path).faults()[0]


def test_report(tmp_path: Path) -> None:
    """The survey states the pair count and the allowance every run, drift or not."""
    lines = pair(tmp_path, DEPLOYED).report()
    assert "1 deployed/validated pair(s) compared" in lines[-1]
    assert "ema" in lines[-1]


def test_strict(tmp_path: Path) -> None:
    """`--strict` drops ema from the allowance, which is how the check graduates."""
    stamp(tmp_path / "arm_ff.pt", DEPLOYED)
    stamp(tmp_path / "arm_s2026.pt", VALIDATED)
    assert TwinDrift(tmp_path, strict=True).faults()
    assert TwinDrift(tmp_path, strict=False).faults() == []
