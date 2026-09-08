"""The three ways a declared control fails, each against the pair on disk that motivated it."""

import json
from pathlib import Path

from kneemri.experiments.control_check import Comparability, Comparison, ControlCheck
from kneemri.runs.run_metrics import RunMetrics

BAND_DENSE_PEAK = 0.8578
"""`band_dense`'s best, and the number four arms declare. Real, because the defect is never the score."""
ADAPT_NULL_PEAK = 0.8240
GOLD_OFFSET = 0.0383
"""How far `rank966_hold`'s declared 0.8956 sits above `band_dense_hold_s2026`'s own curve. A GOLD-ruler
score quoted as a holdout-ruler control — the sign is what separates it from a transcription slip."""


def write_run(directory: Path, name: str, *, peak: float, epochs: int, recipe: dict[str, str]) -> RunMetrics:
    """A metrics file of `epochs` rows ending at `peak`, describing the recipe it was given.

    Line by line rather than through `RunMetrics.record`, which reads the CUDA allocator — a unit test
    may touch nothing it did not create, and the device is not ours.
    """
    path = directory / f"{name}.metrics.jsonl"
    lines = [json.dumps({"recipe": {"epochs": str(epochs)} | recipe})]
    lines += [
        json.dumps({"epoch": epoch, "loss": 1.0, "auc": round(peak - 0.01 * (epochs - 1 - epoch), 4), "per_label": {}})
        for epoch in range(epochs)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RunMetrics(path)


def write_arm(directory: Path, name: str, *, trains: str, control: str, quoted: float) -> Path:
    """An arm script in the repository's own shape: prose, a declaration line, then the `run_train`."""
    path = directory / f"{name}.sh"
    path.write_text(
        "#!/usr/bin/env bash\n"
        f"# {name} — prose that also names {control} without declaring it.\n"
        f"# control  {control} = {quoted:.4f} (best epoch 8), byte-identical recipe with the flag off.\n"
        'source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"\n'
        f"run_train {trains} $RECIPE --epochs 4\n",
        encoding="utf-8",
    )
    return path


def test_declared(tmp_path: Path) -> None:
    """The declaration is read off the `# control` line alone, and the arm off its own `run_train`."""
    check = ControlCheck(tmp_path)
    arm = write_arm(tmp_path, "adapt_null", trains="adapt_null", control="band_dense", quoted=BAND_DENSE_PEAK)
    declared = check.declared(arm)
    assert declared is not None
    assert (declared.arm, declared.control, declared.quoted) == ("adapt_null", "band_dense", BAND_DENSE_PEAK)

    # An arm's prose names several runs. Only the anchored declaration line is a claim about which one
    # the delta is against, which is why the pattern is not a bare search for the word.
    prose = tmp_path / "prose.sh"
    prose.write_text("# it lost to band_dense = 0.8578 by a lot\nrun_train mine\n", encoding="utf-8")
    assert check.declared(prose) is None, "a control named mid-sentence is not a declaration"

    foreign = tmp_path / "foreign.sh"
    foreign.write_text("# control  band_dense = 0.8578\nforeign_run his_script.py\n", encoding="utf-8")
    assert check.declared(foreign) is None, "a foreign arm writes no metrics file, so it has no side to read"


def test_resolve(tmp_path: Path) -> None:
    """Both spellings a control line uses, because an author quotes whichever they had in mind."""
    check = ControlCheck(tmp_path)
    write_run(tmp_path, "band_dense_s2026", peak=BAND_DENSE_PEAK, epochs=12, recipe={})
    seeded = check.resolve("band_dense")
    assert seeded is not None and seeded.name() == "band_dense_s2026", "the seed suffix is resolved"
    assert check.resolve("band_dense_s2026") is not None, "and so is the spelling that carries it"
    assert check.resolve("never_ran") is None


def test_sourced(tmp_path: Path) -> None:
    """A quoted score is measured only if it is on some epoch of the control's own curve."""
    control = write_run(tmp_path, "band_dense", peak=BAND_DENSE_PEAK, epochs=12, recipe={})
    comparability = Comparability()
    assert comparability.sourced(control, BAND_DENSE_PEAK), "the peak is on the curve"
    assert comparability.sourced(control, round(BAND_DENSE_PEAK - 0.03, 4)), "so is a matched earlier epoch"
    assert not comparability.sourced(control, BAND_DENSE_PEAK + GOLD_OFFSET), "a gold-ruler score is on no epoch"
    assert not comparability.sourced(control, 0.8232), "the number `adapt_null` was priced against"


def test_budgeted(tmp_path: Path) -> None:
    """Same asked-for epochs or the delta is a best-of-4 against a best-of-12."""
    comparability = Comparability()
    short = write_run(tmp_path, "adapt_null", peak=ADAPT_NULL_PEAK, epochs=4, recipe={})
    long = write_run(tmp_path, "band_dense", peak=BAND_DENSE_PEAK, epochs=12, recipe={})
    assert not comparability.budgeted(short, long), "4 against 12 is the -0.0338 that was quoted four times"
    assert comparability.budgeted(long, long)


def test_built(tmp_path: Path) -> None:
    """A recipe key present in one run and absent from the other is a code path, not a value."""
    comparability = Comparability()
    newer = write_run(tmp_path, "mine", peak=0.86, epochs=12, recipe={"layer_decay": "0.75", "accumulate": "2"})
    older = write_run(tmp_path, "band_dense", peak=BAND_DENSE_PEAK, epochs=12, recipe={})
    assert comparability.built(newer, older) == ("accumulate", "layer_decay"), "the flags one build never had"
    assert comparability.built(newer, newer) == (), "a run against itself drifts in nothing"


def test_findings(tmp_path: Path) -> None:
    """Every reason at once, and the control's own best beside the score the arm quoted for it."""
    check = ControlCheck(tmp_path)
    write_run(tmp_path, "adapt_null", peak=ADAPT_NULL_PEAK, epochs=4, recipe={"layer_decay": "0.75"})
    write_run(tmp_path, "band_dense", peak=BAND_DENSE_PEAK, epochs=12, recipe={})

    found, measured = check.findings(Comparison("adapt_null", "band_dense", BAND_DENSE_PEAK, tmp_path / "a.sh"))
    assert found == ("BUDGET", "BUILD(1)"), "sourced, but neither the budget nor the build matched"
    assert measured == BAND_DENSE_PEAK

    gold = Comparison("adapt_null", "band_dense", BAND_DENSE_PEAK + GOLD_OFFSET, tmp_path / "a.sh")
    assert check.findings(gold)[0][0] == "UNSOURCED", "the ruler collision leads, because it moves the delta"

    missing = Comparison("adapt_null", "never_ran", BAND_DENSE_PEAK, tmp_path / "a.sh")
    assert check.findings(missing) == (("UNRESOLVED",), None), "a fact about this disk, not about the arm"


def test_audit(tmp_path: Path) -> None:
    """Only the comparisons that cannot carry their delta, worst first — a clean pair is silent."""
    arms = tmp_path / "arms"
    arms.mkdir()
    check = ControlCheck(tmp_path)
    recipe = {"layer_decay": "0.75"}
    write_run(tmp_path, "band_dense", peak=BAND_DENSE_PEAK, epochs=12, recipe=recipe)
    write_run(tmp_path, "adapt_null", peak=ADAPT_NULL_PEAK, epochs=4, recipe=recipe)
    write_run(tmp_path, "clean", peak=0.86, epochs=12, recipe=recipe)

    write_arm(arms, "adapt_null", trains="adapt_null", control="band_dense", quoted=BAND_DENSE_PEAK)
    write_arm(arms, "clean", trains="clean", control="band_dense", quoted=BAND_DENSE_PEAK)
    seen = check.audit(arms)
    assert [row[0].arm for row in seen] == ["adapt_null"], "a same-budget same-build sourced pair is not a finding"
    assert seen[0][1] == ("BUDGET",)


def test_line(tmp_path: Path) -> None:
    """The offset is printed, because it is what tells a transcription slip from a different ruler."""
    comparison = Comparison("rank966_hold", "band_dense_hold_s2026", 0.8956, tmp_path / "rank966_hold.sh")
    line = comparison.line(("UNSOURCED",), 0.8573)
    assert "+0.0383" in line, "a signed offset, so a gold-ruler quote reads as one at a glance"
    assert "rank966_hold.sh" in line, "the declaring script, which is not always the arm's own name"
    assert "no curve" in comparison.line(("UNRESOLVED",), None)
