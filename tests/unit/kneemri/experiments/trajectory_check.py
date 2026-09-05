"""The gate, against the two readings that motivated it and the stitched series that nearly broke it."""

import json
from pathlib import Path

import pytest

from kneemri.experiments.trajectory_check import (
    GATE_EPOCH,
    MIN_CORPUS,
    RunTrajectory,
    Spread,
    TrajectoryGate,
    main,
)
from kneemri.runs.run_metrics import RunMetrics

CEILING = 0.05
"""The largest gain the fixture corpus offers, so a target more than this above an arm is unreachable."""


def write_curve(directory: Path, name: str, scores: list[float], asked: int) -> RunMetrics:
    """A metrics file holding exactly `scores`, describing a recipe that asked for `asked` epochs.

    Written line by line rather than through `RunMetrics.record`, which reads the CUDA allocator: a unit
    test may touch nothing it did not create, and the device is not ours.
    """
    path = directory / f"{name}.metrics.jsonl"
    lines = [json.dumps({"recipe": {"fold": "0", "epochs": str(asked), "patience": "0", "studies": "0"}})]
    lines += [
        json.dumps({"epoch": epoch, "loss": 1.0, "auc": score, "per_label": {}}) for epoch, score in enumerate(scores)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RunMetrics(path)


def climbing(start: float, gain: float, epochs: int = 12) -> list[float]:
    """A curve that reaches `start + gain` after the gate epoch and then holds."""
    return [start - 0.05, start - 0.02, start - 0.01, start] + [start + gain] * (epochs - GATE_EPOCH - 1)


SPAN = (0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, CEILING)
"""The corpus gains: one more than `MIN_CORPUS`, so the median is a member rather than a midpoint."""


def corpus(directory: Path) -> Path:
    """A processed root of twelve-epoch runs whose gate-to-peak gains span 0.01 to `CEILING`."""
    for index, gain in enumerate(SPAN):
        write_curve(directory, f"member_{index}", climbing(0.80, gain), asked=12)
    return directory


def test_scored(tmp_path: Path) -> None:
    """The holdout series, with the NaN rows a dark member writes dropped."""
    path = tmp_path / "dark.metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"recipe": {"epochs": "12"}}),
                json.dumps({"epoch": 0, "auc": 0.81}),
                json.dumps({"epoch": 1, "auc": float("nan")}),
                json.dumps({"epoch": 2, "auc": 0.83}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert TrajectoryGate().scored(RunMetrics(path)) == [0.81, 0.83], "one NaN makes the median and max NaN too"


def test_gain(tmp_path: Path) -> None:
    """The gate-to-peak gain, and the three runs that do not belong in the corpus."""
    gate = TrajectoryGate()

    assert gate.gain(write_curve(tmp_path, "ordinary", climbing(0.80, 0.02), asked=12), 12) == pytest.approx(0.02)

    # A 24-epoch run has twice the anneal left after the gate: median +0.0339 against a 12-epoch +0.0210.
    long_run = write_curve(tmp_path, "nocycle224_long", climbing(0.79, 0.05, epochs=24), asked=24)
    assert gate.gain(long_run, 24) is not None
    assert gate.gain(long_run, 12) is None, "a schedule's headroom may not excuse an arm on another one"

    # lora_r8_v3small_224: 16 rows against a 12-epoch recipe, so its e3 and its peak are different runs.
    # Read as one trajectory it contributes +0.1350, and the maximum is what decides here.
    stitched = write_curve(tmp_path, "lora_r8_v3small_224", climbing(0.69, 0.135, epochs=16), asked=12)
    assert gate.gain(stitched, 12) is None, "more rows than the budget is two runs, not one trajectory"

    short = write_curve(tmp_path, "adapt_null", climbing(0.80, 0.02, epochs=9), asked=12)
    assert gate.gain(short, 12) is None, "a run too short to show its peak understates every longer one"

    # The floor is a SHARE of its OWN budget, not an absolute epoch count. cnx_warm_anneal asked for six
    # and an absolute ten made every short schedule structurally unreadable — a second budget filter
    # wearing a completeness filter's name, refusing the whole warm lane it was built to arbitrate.
    warm = write_curve(tmp_path, "cnx_warm_anneal", climbing(0.80, 0.02, epochs=5), asked=6)
    assert gate.gain(warm, 6) == pytest.approx(0.02), "five of six is as much anneal on record as ten of twelve"
    died = write_curve(tmp_path, "cnx_warm_died", climbing(0.80, 0.02, epochs=4), asked=6)
    assert gate.gain(died, 6) is None, "the share holds at every budget, it does not only move the bar down"


def test_spread(tmp_path: Path) -> None:
    """The distribution, and its refusal to be read off a root too thin to have one."""
    spread = TrajectoryGate().spread(corpus(tmp_path), 12)
    assert spread.runs == len(SPAN)
    assert spread.largest == pytest.approx(CEILING)
    assert spread.median == pytest.approx(0.03)
    assert spread.high == pytest.approx(CEILING)

    thin = tmp_path / "unmounted"
    thin.mkdir()
    with pytest.raises(SystemExit):
        # An unmounted root would otherwise make every arm truncatable against a corpus of nothing.
        TrajectoryGate().spread(thin, 12)

    for index in range(MIN_CORPUS - 1):
        write_curve(thin, f"few_{index}", climbing(0.80, 0.02), asked=12)
    with pytest.raises(SystemExit):
        # The ceiling is the MAXIMUM, and a maximum over seven draws states which seven runs happened to
        # use the schedule, not what a trajectory on it can do. Being wrong in that direction kills a member.
        TrajectoryGate().spread(thin, 12)


def test_standing(tmp_path: Path) -> None:
    """The spread measured from where the arm IS, and its fallback when that epoch admits too few runs."""
    gate = TrajectoryGate()
    root = corpus(tmp_path)

    calibrated = gate.standing(root, 12, GATE_EPOCH)
    assert calibrated.largest == pytest.approx(CEILING), "at the calibrated gate it is the ordinary spread"
    assert gate.standing(root, 12, GATE_EPOCH - 1).largest == pytest.approx(CEILING), "an EARLIER gate never widens"

    # The fixture curves reach their peak at e4 and hold, so a run sitting at e8 has already spent every
    # gain the corpus records. That is the double count `standing` removes: read from e3 the same arm is
    # credited with +0.05 of climb it is standing on top of.
    late = gate.standing(root, 12, 8)
    assert late.gate == 8
    assert late.largest == pytest.approx(0.0), "no run on this schedule gained anything after e8"

    # A gate past the budget matches nothing, and a ceiling read off nothing would truncate every arm.
    fallback = gate.standing(root, 12, 12)
    assert fallback.gate == GATE_EPOCH
    assert fallback.largest == pytest.approx(CEILING), "too thin to decide falls back, it does not refuse"


def test_reading(tmp_path: Path) -> None:
    """One arm against the corpus, read from its LAST epoch so the approximation favours the run."""
    gate = TrajectoryGate()
    spread = gate.spread(corpus(tmp_path), 12)
    aside = tmp_path / "arm"
    aside.mkdir()
    arm = write_curve(aside, "vit_in21k", [0.7239, 0.7541, 0.7751, 0.7755, 0.7847], asked=12)

    reading = gate.reading(arm, 0.8393, spread)
    assert reading.epoch == 4, "the last recorded epoch, not the gate - the arm is credited with the climb it spent"
    assert reading.auc == pytest.approx(0.7847)
    assert reading.ceiling == pytest.approx(0.7847 + CEILING)


def test_truncatable(tmp_path: Path) -> None:
    """The verdict: unreachable only when NO trajectory in the corpus closes the distance."""
    unreachable = RunTrajectory(name="vit_in21k", epoch=4, auc=0.7847, target=0.8393, ceiling=0.7847 + CEILING)
    assert unreachable.truncatable(), "needs +0.0546 against a corpus whose largest gain is smaller"

    # swin_supervised sat at 0.8192 at the gate and was declined LATER, on its finished number. The gate
    # may only truncate, so an arm the corpus could still carry has to survive it.
    open_arm = RunTrajectory(name="swin_supervised", epoch=3, auc=0.8192, target=0.8393, ceiling=0.8192 + CEILING)
    assert not open_arm.truncatable()


def test_spread_line() -> None:
    """The distribution line carries all three numbers, so a truncation can be quoted with its evidence."""
    assert "max +0.0529" in Spread(runs=96, median=0.0210, high=0.0380, largest=0.0529).line()


def test_run_trajectory_line() -> None:
    """The arm's line carries the arithmetic and not only the verdict."""
    stated = RunTrajectory(name="vit_in21k", epoch=4, auc=0.7847, target=0.8393, ceiling=0.8376).line()
    assert "CANNOT REACH" in stated
    assert "needs +0.0546" in stated


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI refuses a run name that is not a path, and a `--run` given no pre-registered target."""
    monkeypatch.setattr("sys.argv", ["trajectory_check", "--run", str(tmp_path / "absent.metrics.jsonl")])
    with pytest.raises(SystemExit) as missing:
        main()
    assert "not on disk" in str(missing.value)

    arm = write_curve(tmp_path, "arm", climbing(0.80, 0.02), asked=12)
    monkeypatch.setattr("sys.argv", ["trajectory_check", "--run", str(arm.path)])
    with pytest.raises(SystemExit) as untargeted:
        main()
    assert "--target" in str(untargeted.value), "reading a target off the curve is fitting the gate to the arm"
