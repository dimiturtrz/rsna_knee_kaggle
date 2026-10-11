"""The three ways a declared control fails, each against the pair on disk that motivated it."""

import json
from pathlib import Path

import numpy as np

from core.findings import TARGETS
from kneemri.evaluation.ledger75 import LedgerSeventyFive
from kneemri.experiments.control_check import Comparability, Comparison, ControlCheck
from kneemri.runs.holdout_record import HoldoutRecord
from kneemri.runs.run_metrics import RunMetrics

BAND_DENSE_PEAK = 0.8578
"""`band_dense`'s best, and the number four arms declare. Real, because the defect is never the score."""
ADAPT_NULL_PEAK = 0.8240
GOLD_OFFSET = 0.0383
"""How far `rank966_hold`'s declared 0.8956 sits above `band_dense_hold_s2026`'s own curve. A GOLD-ruler
score quoted as a holdout-ruler control — the sign is what separates it from a transcription slip. It is
NOT a defect: the arm's own line says `gold`, and gold is the one ruler this side cannot recompute."""
GOLD_DECLARATION = " gold / ~0.857 proxy (same recipe, no flags)."
"""`rank966_hold.sh`'s declaration tail, verbatim. The check reads one keyword out of it rather than
parsing the ruler's spelling, because `L75mid`, `own holdout`, `ledger75` and `896, L75mid` all appear."""


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
    check = ControlCheck(tmp_path, tmp_path)
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
    check = ControlCheck(tmp_path, tmp_path)
    write_run(tmp_path, "band_dense_s2026", peak=BAND_DENSE_PEAK, epochs=12, recipe={})
    seeded = check.resolve("band_dense")
    assert seeded is not None and seeded.name() == "band_dense_s2026", "the seed suffix is resolved"
    assert check.resolve("band_dense_s2026") is not None, "and so is the spelling that carries it"
    assert check.resolve("never_ran") is None

    # THE ARM NAME IS THE THIRD SPELLING. `whole_stack` is an arm whose `--out` writes `wholestack_s2026`,
    # and reading the token as a checkpoint stem put three arms that RAN in the bucket for never-run.
    (tmp_path / "whole_stack.sh").write_text(
        '# control  band_dense = 0.8578\nrun_train whole_stack $RECIPE --out "$MODELS/wholestack_s2026.pt"\n',
        encoding="utf-8",
    )
    write_run(tmp_path, "wholestack_s2026", peak=BAND_DENSE_PEAK, epochs=12, recipe={})
    through_arm = check.resolve("whole_stack")
    assert through_arm is not None and through_arm.name() == "wholestack_s2026", "resolved through --out"

    # AND AN EPOCH EXPORT HAS A HOLDOUT RECORD AND NO METRICS FILE. It is a scoreable run, not a hole:
    # `wsfold0_e6` is `wholestack`'s epoch 6 under a second name, and four arms price against it.
    write_holdout(tmp_path, "wsfold0_e6", studies=60)
    exported = check.resolve("wsfold0_e6")
    assert exported is not None and not exported.rows(), "resolved on its predictions, with no curve"


def write_holdout(directory: Path, name: str, *, studies: int) -> None:
    """A holdout record beside a run, graded so the two cuts actually disagree.

    Truth at 0.68 is the reader's "possible" rung — the one the midpoint counts as a positive and the
    0.75 recut does not. A record of 0s and 1s would score identically under both rulers and prove
    nothing about which one the check read.
    """
    rows = np.arange(studies)
    truth = np.where(rows % 3 == 0, 1.0, np.where(rows % 3 == 1, 0.68, 0.0)).astype(np.float32)
    predictions = np.linspace(0.1, 0.9, studies)
    HoldoutRecord(directory / f"{name}.metrics.jsonl").write(
        [f"study{row}" for row in rows],
        np.repeat(predictions[:, None], len(TARGETS), axis=1),
        np.repeat(truth[:, None], len(TARGETS), axis=1),
        epoch=8,
    )


def test_sourced(tmp_path: Path) -> None:
    """A quoted score is measured if it is on the control's own curve UNDER ANY RULER this side has."""
    control = write_run(tmp_path, "band_dense", peak=BAND_DENSE_PEAK, epochs=12, recipe={})
    comparability = Comparability()
    assert comparability.sourced(control, BAND_DENSE_PEAK), "the peak is on the curve"
    assert comparability.sourced(control, round(BAND_DENSE_PEAK - 0.03, 4)), "so is a matched earlier epoch"
    assert not comparability.sourced(control, 0.8232), "the number `adapt_null` was priced against"
    assert not comparability.sourced(control, BAND_DENSE_PEAK + GOLD_OFFSET), "and no ruler here reaches gold"

    # THE RECUT IS A RULER, NOT A DEFECT. `.metrics.jsonl`'s `auc` is the MIDPOINT macro, and an arm may
    # legitimately declare its control at LEDGER75 — three of them do, and the first version of this
    # check reported all three as fabricated because it knew only the one column.
    write_holdout(tmp_path, "band_dense", studies=60)
    recut = round(LedgerSeventyFive().macro(LedgerSeventyFive().reading(control.checkpoint())), 4)
    assert recut not in {round(float(row["auc"]), 4) for row in control.rows()}, "the cuts disagree here"
    assert comparability.sourced(control, recut), "a control declared at ledger75 is sourced, not invented"


def test_gold(tmp_path: Path) -> None:
    """The one ruler this side cannot compute is read off the declaration, by keyword and not by parse."""
    declared = Comparison("rank966_hold", "band_dense_hold", 0.8956, tmp_path / "a.sh", GOLD_DECLARATION)
    assert declared.gold(), "the arm names its ruler on its own line"
    assert not Comparison("adapt_null", "band_dense", BAND_DENSE_PEAK, tmp_path / "a.sh").gold()
    assert not Comparison("a", "b", 0.85, tmp_path / "a.sh", " (own holdout, L75mid, n=909)").gold()


def test_scoreable(tmp_path: Path) -> None:
    """A run this side can price off its own predictions, which is what makes a curve-less export real."""
    comparability = Comparability()
    exported = write_run(tmp_path, "wsfold0_e6", peak=BAND_DENSE_PEAK, epochs=0, recipe={})
    assert not comparability.scoreable(exported), "no kept predictions, so no ruler reaches it"
    write_holdout(tmp_path, "wsfold0_e6", studies=60)
    assert comparability.scoreable(exported), "the holdout record is the producer, not the metrics file"


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
    check = ControlCheck(tmp_path, tmp_path)
    write_run(tmp_path, "adapt_null", peak=ADAPT_NULL_PEAK, epochs=4, recipe={"layer_decay": "0.75"})
    write_run(tmp_path, "band_dense", peak=BAND_DENSE_PEAK, epochs=12, recipe={})

    found, measured = check.findings(Comparison("adapt_null", "band_dense", BAND_DENSE_PEAK, tmp_path / "a.sh"))
    assert found == ("BUDGET", "BUILD(1)"), "sourced, but neither the budget nor the build matched"
    assert measured == BAND_DENSE_PEAK

    off_curve = Comparison("adapt_null", "band_dense", BAND_DENSE_PEAK + GOLD_OFFSET, tmp_path / "a.sh")
    assert check.findings(off_curve)[0][0] == "UNSOURCED", "a score on no ruler leads, because it moves the delta"

    gold = Comparison("adapt_null", "band_dense", BAND_DENSE_PEAK + GOLD_OFFSET, tmp_path / "a.sh", GOLD_DECLARATION)
    assert check.findings(gold)[0][0] == "UNRULED", "the arm declared a ruler this side cannot recompute"

    missing = Comparison("adapt_null", "never_ran", BAND_DENSE_PEAK, tmp_path / "a.sh")
    assert check.findings(missing) == (("UNRESOLVED",), None), "a control nothing on disk produced"

    # THE OPPOSITE READING, and merging the two said neither: an arm queued behind another arm has no
    # metrics file either, and reporting that as an unfindable control put 37 waiting arms in the same
    # bucket as five write-ups citing a run that does not exist.
    queued = Comparison("not_yet", "band_dense", BAND_DENSE_PEAK, tmp_path / "a.sh")
    assert check.findings(queued) == (("UNRUN",), None), "an arm that has not spent the card is not a defect"


def test_audit(tmp_path: Path) -> None:
    """Only the comparisons that cannot carry their delta, worst first — a clean pair is silent."""
    arms = tmp_path / "arms"
    arms.mkdir()
    check = ControlCheck(tmp_path, arms)
    recipe = {"layer_decay": "0.75"}
    write_run(tmp_path, "band_dense", peak=BAND_DENSE_PEAK, epochs=12, recipe=recipe)
    write_run(tmp_path, "adapt_null", peak=ADAPT_NULL_PEAK, epochs=4, recipe=recipe)
    write_run(tmp_path, "clean", peak=0.86, epochs=12, recipe=recipe)

    write_arm(arms, "adapt_null", trains="adapt_null", control="band_dense", quoted=BAND_DENSE_PEAK)
    write_arm(arms, "clean", trains="clean", control="band_dense", quoted=BAND_DENSE_PEAK)
    seen = check.audit()
    assert [row[0].arm for row in seen] == ["adapt_null"], "a same-budget same-build sourced pair is not a finding"
    assert seen[0][1] == ("BUDGET",)


def test_summary(tmp_path: Path) -> None:
    """Each finding counted once per comparison, and BUILD by its prefix since it carries a count."""
    check = ControlCheck(tmp_path, tmp_path)
    arm = Comparison("a", "b", BAND_DENSE_PEAK, tmp_path / "a.sh")
    seen: list[tuple[Comparison, tuple[str, ...], float | None]] = [
        (arm, ("UNSOURCED", "BUILD(2)"), None),
        (arm, ("BUDGET", "BUILD(1)"), 0.85),
        (arm, ("UNRUN",), None),
    ]

    assert check.summary(seen) == (
        "3 declared comparisons flagged: 1 unsourced, 0 unruled, 1 budget, 2 build, 0 unresolved, 1 unrun"
    )
    assert check.summary([]).startswith("0 declared comparisons flagged: 0 unsourced")


def test_line(tmp_path: Path) -> None:
    """The offset is printed, because it is what tells a transcription slip from a different ruler."""
    comparison = Comparison("rank966_hold", "band_dense_hold_s2026", 0.8956, tmp_path / "rank966_hold.sh")
    line = comparison.line(("UNSOURCED",), 0.8573)
    assert "+0.0383" in line, "a signed offset, so a gold-ruler quote reads as one at a glance"
    assert "rank966_hold.sh" in line, "the declaring script, which is not always the arm's own name"
    assert "no curve" in comparison.line(("UNRESOLVED",), None)


def test_recorded(tmp_path: Path) -> None:
    """An unreadable recipe is silence about the build, never a drift in every key it cannot see."""
    comparability = Comparability()
    run = write_run(tmp_path, "band_dense", peak=BAND_DENSE_PEAK, epochs=12, recipe={"layer_decay": "0.75"})
    write_holdout(tmp_path, "wsfold0_e6", studies=60)
    exported = RunMetrics(tmp_path / "wsfold0_e6.pt")
    assert comparability.recorded(run, run)
    assert not comparability.recorded(run, exported), "an epoch export writes predictions, not a recipe"
    assert comparability.built(run, exported) == ("epochs", "layer_decay"), "which is why built() may not be asked"
