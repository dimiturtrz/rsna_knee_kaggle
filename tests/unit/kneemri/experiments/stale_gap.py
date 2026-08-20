"""The patience derivation, checked on curves whose answer is known by construction."""

import json
from pathlib import Path

from kneemri.experiments.stale_gap import METRICS_SUFFIX, StaleGaps, StaleRun


def _write(directory: Path, name: str, scores: list[float]) -> Path:
    """A metrics file with a recipe header and one line per scored epoch, as a run writes it."""
    lines = [json.dumps({"recipe": {"epochs": str(len(scores))}})]
    lines += [json.dumps({"epoch": epoch, "auc": score}) for epoch, score in enumerate(scores)]
    path = directory / f"{name}{METRICS_SUFFIX}"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_of(tmp_path: Path):
    """A curve that sags for two epochs and then beats its own best reports a survived stall of two."""
    run = StaleRun.of(_write(tmp_path, "sagging", [0.70, 0.80, 0.79, 0.78, 0.85, 0.84]))

    assert run is not None
    assert run.name == "sagging"
    assert run.best_epoch == 4, "the best is the peak, not the last epoch"
    assert run.survived == 2, "epochs 2 and 3 did not improve and epoch 4 recovered from them"
    assert run.wasted == 1, "one epoch ran after the best and its checkpoint was discarded"


def test_of_unscored(tmp_path: Path):
    """A recipe-only file is skipped rather than raised on — a fold -1 fit writes exactly that."""
    path = tmp_path / f"final{METRICS_SUFFIX}"
    path.write_text(json.dumps({"recipe": {"epochs": "12"}}), encoding="utf-8")

    assert StaleRun.of(path) is None


def test_runs(tmp_path: Path):
    """Runs come back worst-wasting first, so the prize sits at the top of the table."""
    _write(tmp_path, "tight", [0.70, 0.85])
    _write(tmp_path, "loose", [0.85, 0.70, 0.70, 0.70])

    names = [run.name for run in StaleGaps(tmp_path).runs()]

    assert names == ["loose", "tight"], "three discarded epochs outrank none"


def test_lines(tmp_path: Path):
    """The report names the run that set the floor, because a bare number cannot be argued with."""
    _write(tmp_path, "late", [0.70, 0.69, 0.69, 0.69, 0.90])
    _write(tmp_path, "early", [0.90, 0.70])

    lines = StaleGaps(tmp_path).lines()

    assert "longest recovered stall: 3 epochs (late)" in lines[-3]
    assert "must exceed it" in lines[-3], "the value is a floor, not the setting itself"
    assert "epochs run past a best: 1" in lines[-2], "late peaked last and wasted none; early wasted one"


def test_saving(tmp_path: Path):
    """A stopper sits through its own patience first, so the saving is the waste MINUS the value."""
    _write(tmp_path, "long", [0.90, 0.7, 0.7, 0.7, 0.7, 0.7])
    _write(tmp_path, "short", [0.90, 0.7])

    gaps = StaleGaps(tmp_path)

    assert gaps.saving(0) == 6, "with no patience every epoch past a best is recoverable"
    assert gaps.saving(2) == 3, "long gives back two of its five, short's one is inside the value"
    assert gaps.saving(9) == 0, "a patience longer than any tail buys nothing at all"


def test_lines_empty(tmp_path: Path):
    """An empty directory says so instead of dividing by no runs."""
    assert StaleGaps(tmp_path).lines() == [f"no metrics files under {tmp_path}"]
