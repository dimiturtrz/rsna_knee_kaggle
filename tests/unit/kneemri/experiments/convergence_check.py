"""The three verdicts, each against the run on disk that motivated it."""

import json
from pathlib import Path

import pytest

from kneemri.experiments.convergence_check import Convergence, ConvergenceCheck, RunConvergence, main
from kneemri.runs.run_metrics import RunMetrics

SETTLED_PEAK = 6
"""The peak `v3base_dense` and `raddino_dense` SHARE. Every fixture here peaks at the same epoch on
purpose: it is the pair that proves the score cannot tell a measurement from a floor."""


def write_run(
    directory: Path,
    name: str,
    *,
    peak: int,
    epochs: int,
    asked: int,
    fold: int = 0,
    patience: int = 0,
    first: int = 0,
) -> RunMetrics:
    """A metrics file with `epochs` rows peaking at `peak`, describing a recipe that asked for `asked`.

    Written line by line rather than through `RunMetrics.record`, which reads the CUDA allocator — a
    unit test may touch nothing it did not create, and the device is not ours.

    `first` is the epoch the rows START at, which is what a `--resume` round looks like on disk: the
    earlier rounds have been rotated aside and this file's own epochs go on counting from where they
    stopped.
    """
    path = directory / f"{name}.metrics.jsonl"
    recipe = {"fold": str(fold), "epochs": str(asked), "patience": str(patience)}
    lines = [json.dumps({"recipe": recipe})]
    lines += [
        json.dumps({"epoch": epoch, "loss": 1.0, "auc": 0.80 + (0.01 if epoch == peak else 0.0), "per_label": {}})
        for epoch in range(first, first + epochs)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RunMetrics(path)


def test_verdict(tmp_path: Path) -> None:
    """The three outcomes, and the pair the score alone cannot separate."""
    check = ConvergenceCheck()

    # v3base_dense: twelve asked, twelve run, peaked at six and never beat it again.
    converged = check.verdict(write_run(tmp_path, "v3base_dense", peak=SETTLED_PEAK, epochs=12, asked=12))
    assert converged.verdict is Convergence.CONVERGED
    assert converged.best_epoch == SETTLED_PEAK
    assert converged.epochs_done == 12

    # raddino_dense: twelve asked, SEVEN run, same peak epoch. `ARM FAILED (trainer exit 1)`.
    truncated = check.verdict(write_run(tmp_path, "raddino_dense", peak=SETTLED_PEAK, epochs=7, asked=12))
    assert truncated.verdict is Convergence.TRUNCATED
    assert truncated.best_epoch == converged.best_epoch, "the peak epoch is identical - only finishing differs"
    assert truncated.epochs_asked == 12

    # res392_lr8e-6_ws: twelve asked, TEN run, same shape on disk as raddino — but patience=3 armed.
    # Without this case the check's own first run reported the resolution refutation as a crash.
    stopped = check.verdict(write_run(tmp_path, "res392", peak=SETTLED_PEAK, epochs=10, asked=12, patience=3))
    assert stopped.verdict is Convergence.STOPPED
    assert stopped.settled(), "a stopper firing is a decision, so the score is the recipe's number"

    # cnx_dense: twelve asked, twelve run, peak ON the last one with the loss still falling.
    undertrained = check.verdict(write_run(tmp_path, "cnx_dense", peak=11, epochs=12, asked=12))
    assert undertrained.verdict is Convergence.UNDERTRAINED

    # A dark member holds nothing back, so it has no epoch to peak on and cannot be judged.
    dark = check.verdict(write_run(tmp_path, "base_dense_ff", peak=SETTLED_PEAK, epochs=12, asked=12, fold=-1))
    assert dark.verdict is Convergence.UNSCORED
    assert dark.best_epoch is None


def write_plateau(directory: Path, name: str, tail: list[float]) -> RunMetrics:
    """A twelve-epoch run whose last four scores are given, the rest flat below them."""
    path = directory / f"{name}.metrics.jsonl"
    scores = [tail[0]] * (12 - len(tail)) + tail
    lines = [json.dumps({"recipe": {"fold": "0", "epochs": "12", "patience": "0"}})]
    lines += [
        json.dumps({"epoch": epoch, "loss": 1.0, "auc": auc, "per_label": {}}) for epoch, auc in enumerate(scores)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RunMetrics(path)


def test_rising(tmp_path: Path) -> None:
    """A budget binds by SLOPE, not by argmax — the plateau that ends high is finished."""
    check = ConvergenceCheck()

    # mixup_dense, off disk: the highest point is the last one and the run gained 0.0004 reaching it.
    plateau = write_plateau(tmp_path, "mixup_dense", [0.8548, 0.8541, 0.8546, 0.8552])
    assert not check.rising(plateau)
    assert check.verdict(plateau).verdict is Convergence.CONVERGED, "a plateau argues for no longer budget"

    # adapt_null, off disk: four epochs, still climbing 0.078 when the budget ended.
    climbing = write_plateau(tmp_path, "adapt_null", [0.7451, 0.7880, 0.8180, 0.8240])
    assert check.rising(climbing)
    assert check.verdict(climbing).verdict is Convergence.UNDERTRAINED


def test_asked(tmp_path: Path) -> None:
    """The budget comes from the recipe, and is None when the file predates recipes."""
    check = ConvergenceCheck()
    assert check.asked(write_run(tmp_path, "with_recipe", peak=1, epochs=4, asked=12)) == 12

    bare = tmp_path / "no_recipe.metrics.jsonl"
    bare.write_text(json.dumps({"epoch": 0, "loss": 1.0, "auc": 0.8, "per_label": {}}) + "\n", encoding="utf-8")
    assert check.asked(RunMetrics(bare)) is None


def test_stoppable(tmp_path: Path) -> None:
    """`patience=0` is the default and means no stopper was armed."""
    check = ConvergenceCheck()
    assert check.stoppable(write_run(tmp_path, "with_stopper", peak=1, epochs=10, asked=12, patience=3))
    assert not check.stoppable(write_run(tmp_path, "no_stopper", peak=1, epochs=10, asked=12, patience=0))


def test_smoke(tmp_path: Path) -> None:
    """A capped-corpus run is short by construction and is not a truncation."""
    check = ConvergenceCheck()
    full = write_run(tmp_path, "full_corpus", peak=1, epochs=4, asked=12)
    assert not check.smoke(full)

    smoked = tmp_path / "member_fold0.metrics.jsonl"
    smoked.write_text(json.dumps({"recipe": {"fold": "0", "epochs": "1", "studies": "60"}}) + "\n", encoding="utf-8")
    assert check.smoke(RunMetrics(smoked))


def test_benchmark(tmp_path: Path) -> None:
    """A one-epoch timing probe cannot peak anywhere but its last epoch, so it earns no verdict."""
    check = ConvergenceCheck()
    probe = write_run(tmp_path, "throughput_bf16", peak=0, epochs=1, asked=1)
    assert check.benchmark(probe)
    assert not check.benchmark(write_run(tmp_path, "real_run", peak=6, epochs=12, asked=12))

    audited = [run.name for run in check.audit(tmp_path)]
    assert "throughput_bf16" not in audited
    assert "real_run" in audited


def test_audit(tmp_path: Path) -> None:
    """Every scored run under a root, smokes dropped, unsettled first."""
    write_run(tmp_path, "v3base_dense", peak=SETTLED_PEAK, epochs=12, asked=12)
    write_run(tmp_path, "raddino_dense", peak=SETTLED_PEAK, epochs=7, asked=12)
    (tmp_path / "member_fold0.metrics.jsonl").write_text(
        json.dumps({"recipe": {"fold": "0", "epochs": "1", "studies": "60"}}) + "\n", encoding="utf-8"
    )
    (tmp_path / "never_scored.metrics.jsonl").write_text(
        json.dumps({"recipe": {"fold": "0", "epochs": "12"}}) + "\n", encoding="utf-8"
    )

    seen = ConvergenceCheck().audit(tmp_path)

    assert [run.name for run in seen] == ["raddino_dense", "v3base_dense"], "unsettled sorts first"
    assert seen[0].verdict is Convergence.TRUNCATED


def test_settled() -> None:
    """Only the two ways of ending on purpose, plus the unscorable, may be quoted as the recipe's number."""
    settled = {
        verdict
        for verdict in Convergence
        if RunConvergence("run", verdict, best_epoch=1, epochs_done=12, epochs_asked=12).settled()
    }
    assert settled == {Convergence.CONVERGED, Convergence.STOPPED, Convergence.UNSCORED}


def test_line() -> None:
    """The line carries the evidence, not only the verdict."""
    line = RunConvergence("raddino_dense", Convergence.TRUNCATED, 6, 7, 12).line()
    assert "TRUNCATED" in line
    assert "raddino_dense" in line
    assert "best e6 of 7 run, 12 asked" in line


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One run at a time from the CLI, exiting non-zero only for a truncation."""
    truncated = write_run(tmp_path, "raddino_dense", peak=SETTLED_PEAK, epochs=7, asked=12)
    monkeypatch.setattr("sys.argv", ["convergence_check", "--run", str(truncated.path)])
    with pytest.raises(SystemExit) as exit_code:
        main()
    assert exit_code.value.code == 1

    converged = write_run(tmp_path, "v3base_dense", peak=SETTLED_PEAK, epochs=12, asked=12)
    monkeypatch.setattr("sys.argv", ["convergence_check", "--run", str(converged.path)])
    main()

    # A run NAME resolves to a path that is not there, and reading no file used to return UNSCORED —
    # a verdict about a real state that hides a typo, which read a converged run as having no score.
    monkeypatch.setattr("sys.argv", ["convergence_check", "--run", "v3base_dense_s2026"])
    with pytest.raises(SystemExit, match="not on disk"):
        main()


def test_reached(tmp_path: Path) -> None:
    """Epochs GOT THROUGH, which is the row count only for a run that started at epoch zero."""
    check = ConvergenceCheck()

    whole = write_run(tmp_path, "whole", peak=SETTLED_PEAK, epochs=12, asked=12)
    assert check.reached(whole.rows()) == 12

    # dml_a_s2026: 24 epochs across three --resume rounds, of which this file holds the last eight.
    resumed = write_run(tmp_path, "resumed", peak=18, epochs=8, asked=24, first=16)
    assert check.reached(resumed.rows()) == 24
    assert len(resumed.rows()) == 8

    assert check.reached([]) == 0


def test_resumed_run_is_not_truncated(tmp_path: Path) -> None:
    """A finished run resumed in rounds is CONVERGED, not a floor.

    The verdict this check exists to protect ran into its own blind spot: `dml_a_s2026` completed all
    24 epochs and read TRUNCATED, whose stated meaning is that the anneal never ran. It was carrying a
    pre-registered branch that closes a family, so the false positive was one step from retracting a
    real result.
    """
    check = ConvergenceCheck()
    resumed = check.verdict(write_run(tmp_path, "dml_a_s2026", peak=18, epochs=8, asked=24, first=16))
    assert resumed.verdict is Convergence.CONVERGED
    assert resumed.epochs_done == 24
    assert resumed.settled()

    # And a round that died mid-budget is still a floor: the fix reads the epoch, not the rotation.
    killed = check.verdict(write_run(tmp_path, "died", peak=18, epochs=4, asked=24, first=16))
    assert killed.verdict is Convergence.TRUNCATED
    assert killed.epochs_done == 20
