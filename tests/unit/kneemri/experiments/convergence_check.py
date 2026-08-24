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
    directory: Path, name: str, *, peak: int, epochs: int, asked: int, fold: int = 0, patience: int = 0
) -> RunMetrics:
    """A metrics file with `epochs` rows peaking at `peak`, describing a recipe that asked for `asked`.

    Written line by line rather than through `RunMetrics.record`, which reads the CUDA allocator — a
    unit test may touch nothing it did not create, and the device is not ours.
    """
    path = directory / f"{name}.metrics.jsonl"
    recipe = {"fold": str(fold), "epochs": str(asked), "patience": str(patience)}
    lines = [json.dumps({"recipe": recipe})]
    lines += [
        json.dumps({"epoch": epoch, "loss": 1.0, "auc": 0.80 + (0.01 if epoch == peak else 0.0), "per_label": {}})
        for epoch in range(epochs)
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
