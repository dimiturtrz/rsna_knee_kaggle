import json
from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.run_comparison import SUPERVISION_AXES, VIEW_AXES, PairedWidths, RunComparison
from kneemri.evaluation.supervision_gap import Bottleneck
from kneemri.runs.holdout_record import DERIVED, TRAINED, HoldoutRecord
from kneemri.runs.run_record import EpochRow

STUDIES = 40

CONTROL: list[EpochRow] = [
    {"epoch": 0, "loss": 1.0, "auc": 0.80, "seconds": 10, "per_label": {"MCL": 0.70, "Effusion": 0.90}},
    {"epoch": 1, "loss": 0.9, "auc": 0.84, "seconds": 10, "per_label": {"MCL": 0.75, "Effusion": 0.93}},
]
ARM: list[EpochRow] = [
    {"epoch": 0, "loss": 1.0, "auc": 0.81, "seconds": 10, "per_label": {"MCL": 0.72, "Effusion": 0.90}},
    {"epoch": 1, "loss": 0.8, "auc": 0.86, "seconds": 10, "per_label": {"MCL": 0.85, "Effusion": 0.87}},
]


def _run(tmp_path: Path, name: str, rows: list[EpochRow]) -> Path:
    checkpoint = tmp_path / f"{name}.pt"
    checkpoint.with_suffix(".metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return checkpoint


def _describe(checkpoint: Path, recipe: dict[str, str]) -> None:
    """Prepend a recipe line, the way a real run writes one before its first epoch."""
    path = checkpoint.with_suffix(".metrics.jsonl")
    path.write_text(json.dumps({"recipe": recipe}) + "\n" + path.read_text(encoding="utf-8"), encoding="utf-8")


def test_deltas(tmp_path: Path):
    """Deltas are taken against the control's BEST epoch, per label, and the mean comes with them.

    The per-label split is the whole point: this arm gains 0.10 on MCL and LOSES 0.06 on Effusion,
    which is a concentrated gain — the shape the resolution and extended-slots arms predict — and a
    mean of +0.02 alone cannot tell that apart from a uniform lift.
    """
    comparison = RunComparison(_run(tmp_path, "ctl", CONTROL), [_run(tmp_path, "arm", ARM)])
    deltas = comparison.deltas()["arm.metrics"]
    assert deltas["MCL"] == pytest.approx(0.10)
    assert deltas["Effusion"] == pytest.approx(-0.06)
    assert deltas["MEAN"] == pytest.approx(0.02)


def test_deltas_refuses_a_control_with_no_scored_epoch(tmp_path: Path):
    """A final fit records NaN AUCs, so it can be an ARM but never the control it is measured against."""
    empty: dict[str, float] = {}
    unscored: list[EpochRow] = [{"epoch": 0, "loss": 1.0, "auc": float("nan"), "seconds": 10, "per_label": empty}]
    comparison = RunComparison(_run(tmp_path, "final", unscored), [_run(tmp_path, "arm", ARM)])
    with pytest.raises(ValueError, match="no scored epoch"):
        comparison.deltas()


def test_report(tmp_path: Path):
    """The report names the control's best epoch and orders labels by the size of the move."""
    comparison = RunComparison(_run(tmp_path, "ctl", CONTROL), [_run(tmp_path, "arm", ARM)])
    report = comparison.report()
    assert "0.8400" in report and "epoch 1" in report
    assert report.index("MCL") < report.index("Effusion"), "the widest delta is read first"
    assert "by structure scale" in report, "the crop gate is a pattern across labels, not twelve numbers by eye"


def test_report_names_a_stalled_epoch(tmp_path: Path):
    """A stalled arm says so IN the comparison, because this is where its cost gets read as a result.

    `slices16_ws_s2026`'s shape: one 30,684 s epoch among epochs of ~440. Read from the total alone the
    arm looks 8.8x its control; its real per-epoch cost is 1.3x, and the ledger's "33% slower" was right
    while the arithmetic contradicting it was the broken instrument.
    """
    empty: dict[str, float] = {}
    rows: list[EpochRow] = [
        {"epoch": e, "loss": 1.0, "auc": 0.8, "seconds": s, "per_label": empty, "started": 1_760_000_000 + e}
        for e, s in enumerate([440, 30684, 445, 438])
    ]
    comparison = RunComparison(_run(tmp_path, "ctl", CONTROL), [_run(tmp_path, "slow", rows)])
    report = comparison.report()
    assert "STALLED epoch 1" in report and "30684s" in report
    assert "2025-10-09" in report, "the clock time is what lines the stall up against a suspend"
    assert "STALLED epoch 0" not in report, "a 440 s epoch among 440 s epochs is a run, not a stall"


def test_report_names_what_else_moved(tmp_path: Path):
    """A delta is a claim about ONE axis. What the arm changed besides it decides whether that holds."""
    control = _run(tmp_path, "ctl", CONTROL)
    arm = _run(tmp_path, "arm", ARM)
    _describe(control, {"crop_mm": "130.0", "centre": "image", "epochs": "8"})
    _describe(arm, {"crop_mm": "80.0", "centre": "tissue", "epochs": "8"})

    report = RunComparison(control, [arm]).report()
    assert "crop_mm 130.0→80.0" in report
    assert "centre image→tissue" in report
    assert "epochs" not in report.split("vs control:")[1], "an option both runs share is not a difference"


def _stamp(checkpoint: Path, config: dict[str, object]) -> None:
    """The config block a run writes into its checkpoint — the reading, as both eras record it."""
    import torch  # noqa: PLC0415 - only the stamped-checkpoint tests need it

    torch.save({"options": {"epochs": "12"}, "config": config, "epoch": 1, "holdout": 0.84}, checkpoint)


def test_axes_name_options_that_still_exist():
    """Both axis lists are spelled by hand, and a renamed flag would empty one of them in silence.

    The default of the supervision split is MODEL, so `SUPERVISION_AXES` going stale does not raise: it
    just judges every reader arm on the six findings it cannot move, which is exactly the misreading this
    module was fixed for twice on 2026-08-17. `img` is the one deliberate exception — the view axes are
    spelled as the checkpoint's CONFIG spells them, and that record has no flag of the same name.
    """

    from kneemri.training.config_tree import TrainConfig  # noqa: PLC0415 - a test-only reach across layers

    recorded = {leaf.name for leaf, _ in TrainConfig.leaves()}

    assert SUPERVISION_AXES.issubset(recorded), "a supervision axis no run records is one no arm can be judged on"
    assert (VIEW_AXES - {"img"}).issubset(recorded), "the view axes are option names too, apart from the config's own"


def test_disagreements():
    """Only a key BOTH records carry can differ; one the control predates is a question, not a move.

    The absence-as-difference reading is what let a schedule arm and a loss arm each print a resolution
    verdict and a `READER change` they had not earned, on 2026-08-17.
    """
    control = {"crop_mm": "130.0", "epochs": "12"}

    assert RunComparison.disagreements(control, {"crop_mm": "80.0", "epochs": "12"}) == {"crop_mm": "80.0"}
    predates = RunComparison.disagreements(control, {"epochs": "12", "centre": "tissue"})
    assert predates == {}, "an option only the arm recorded is a question, not a difference"
    assert RunComparison.disagreements(control, control) == {}


def test_switched_on():
    """The one absence that IS a difference: a flag the arm turned on that no control of any era had.

    `calibrated_targets_s2026` printed as a MODEL change judged on the model-bottlenecked six, although
    it is the project's canonical supervision arm — because the flag is younger than its control's
    recipe, so `disagreements` dropped it. Off reads in two spellings, boolean and numeric, and both are
    still off.
    """
    control = {"crop_mm": "130.0"}
    theirs = {"crop_mm": "130.0", "calibrated_targets": "True", "train_size": "0", "per_label_loss": "False"}

    assert RunComparison.switched_on(control, theirs) == {"calibrated_targets"}
    assert RunComparison.switched_on(control, {"crop_mm": "80.0"}) == set(), "a shared key is disagreements' job"
    assert RunComparison.switched_on(control, {"lora_rank": "8"}) == {"lora_rank"}, "a rate off zero is on"


def test_view(tmp_path: Path):
    """The reading comes from the config block, which records it for a control whose FLAGS do not.

    `wsfold0_e6` is that control: its options block names no crop at all, while its config states
    `crop_mm 130.0`. Read off the recipe, every crop arm measured against it looked unjudgeable.
    """
    control, bare = _run(tmp_path, "ctl", CONTROL), _run(tmp_path, "bare", ARM)
    _stamp(control, {"img": 336, "crop_mm": 130.0})

    comparison = RunComparison(control, [bare])

    assert comparison.view(comparison.control) == {"img": "336", "crop_mm": "130.0"}
    assert comparison.view(comparison.arms[0]) is None, "a run with no checkpoint states no reading"


def test_moved_view(tmp_path: Path):
    """Only an arm that changed the field of view earns a verdict about the field of view.

    The 2026-08-17 re-read printed one under a pair-memory arm, telling a loss change that resolution
    was not its bottleneck — an axis that arm held fixed. The crop arm is the other half of the same
    gate: it moved the view against a control whose flag list predates `crop_mm`, and must still be
    judged, which is why this is read off the config and not the recipe.
    """
    control = _run(tmp_path, "ctl", CONTROL)
    view, loss, silent = (_run(tmp_path, name, ARM) for name in ("view", "loss", "silent"))
    _stamp(control, {"img": 336, "crop_mm": 130.0})
    _stamp(view, {"img": 336, "crop_mm": 80.0, "centre": "tissue"})
    _stamp(loss, {"img": 336, "crop_mm": 130.0, "pair_memory": True})

    comparison = RunComparison(control, [view, loss, silent])

    assert comparison.moved_view(comparison.arms[0]), "a tighter crop is a change of view"
    assert not comparison.moved_view(comparison.arms[1]), "a loss change says nothing about the view"
    assert not comparison.moved_view(comparison.arms[2]), "an unrecorded reading is not evidence of a move"


def test_moved(tmp_path: Path):
    """Every option that differs, from the recipe AND from the view the two configs record.

    Both sources, because a crop arm against a control whose flag list predates `crop_mm` moved the
    view while neither recipe can say so — and the kind of change and the half it may be judged on are
    two readings of this one set, which is why it has one home.
    """
    control, arm, silent = (_run(tmp_path, name, CONTROL) for name in ("ctl", "arm", "silent"))
    younger = _run(tmp_path, "younger", CONTROL)
    _describe(control, {"per_label_loss": "False"})
    _describe(arm, {"per_label_loss": "True"})
    _describe(younger, {"per_label_loss": "False", "calibrated_targets": "True", "patience": "3"})
    _stamp(control, {"img": 336, "crop_mm": 130.0})
    _stamp(arm, {"img": 336, "crop_mm": 80.0})

    comparison = RunComparison(control, [arm, silent, younger])

    assert comparison.moved(comparison.arms[0]) == {"per_label_loss", "crop_mm"}
    assert comparison.moved(comparison.arms[1]) is None, "an unrecorded recipe is not an empty diff"
    assert comparison.moved(comparison.arms[2]) == {"calibrated_targets", "patience"}, "a flag the control predates"


def test_judged_on(tmp_path: Path):
    """A reader change is restricted to the label half, an imaging change to the model half.

    The point of the restriction: averaged over all twelve, either change is diluted by the six it
    cannot affect, on a ruler whose floor is 0.0053. The line says it is the arm's CLAIM and not its
    score, because the ledger number has to stay the twelve-label mean to stay comparable with itself.
    """
    split = Bottleneck(label_side=("Effusion",), model_side=("MCL",))
    control, reader, imaging = (_run(tmp_path, name, CONTROL) for name in ("ctl", "reader", "imaging"))
    _describe(control, {"per_label_loss": "False", "img": "336"})
    _describe(reader, {"per_label_loss": "True", "img": "336"})
    _describe(imaging, {"per_label_loss": "False", "img": "448"})

    comparison = RunComparison(control, [reader, imaging], bottleneck=split)
    deltas = {"MCL": 0.10, "Effusion": -0.06, "MEAN": 0.02}

    read = "\n".join(comparison.judged_on(comparison.arms[0], deltas))
    assert "on its own 1 label-bottlenecked: mean -0.0600 (Effusion)" in read
    assert "not its score" in read, "the headline stays the twelve-label mean"
    assert "on its own 1 model-bottlenecked: mean +0.1000 (MCL)" in "\n".join(
        comparison.judged_on(comparison.arms[1], deltas)
    )
    assert RunComparison(control, [reader]).judged_on(comparison.arms[0], deltas) == [], "no split, no restriction"


def test_report_names_an_arm_it_could_not_read(tmp_path: Path):
    """A path matching no run is NAMED, not dropped — a short table with no reason reads as a null arm.

    `--arms intensity392_...` (the file is `intensity392_ws_...`) printed the control and nothing else.
    """
    report = RunComparison(_run(tmp_path, "ctl", CONTROL), [tmp_path / "absent.pt"]).report()

    assert "absent.metrics  NOT READ: does not exist" in report


def test_unmatched(tmp_path: Path):
    """Only the arm recorded it, so the control ran the default of the day — a question, not a delta.

    Kept apart because the two need different judgements: `train_size` was added after most of the
    ledger ran and is a no-op, while a recorded `backbone_lr` that disagrees is a second axis.
    """
    control = _run(tmp_path, "ctl", CONTROL)
    arm = _run(tmp_path, "arm", ARM)
    _describe(control, {"crop_mm": "130.0"})
    _describe(arm, {"crop_mm": "80.0", "train_size": "0"})

    comparison = RunComparison(control, [arm])
    said = comparison.unmatched(comparison.arms[0])

    assert "crop_mm 130.0→80.0" in said
    assert "control predates train_size=0" in said
    assert "unset" not in said, "an absence is not a value the control held"

    # The throughput flags are not axes: the `throughput` arm measured all four combinations at 0.7601
    # to 0.7612 on one fold, a spread five times under the noise floor. Calling them confounds made
    # every older run unmatched against every newer one on a difference the instrument cannot see.
    older, newer = _run(tmp_path, "older", CONTROL), _run(tmp_path, "newer", ARM)
    _describe(older, {"crop_mm": "130.0", "compile_encoder": "False", "recompute_activations": "False"})
    _describe(newer, {"crop_mm": "130.0", "compile_encoder": "True", "recompute_activations": "True"})
    again = RunComparison(older, [newer])

    assert again.unmatched(again.arms[0]) == "matched on every option both runs recorded"


def test_recipe(tmp_path: Path):
    """A control that wrote no metrics file still recorded its recipe — in its checkpoint.

    `wsfold0_e6` is that run and it is the control knee-bot's finding rests on. Reading the metrics
    file alone reported the pair as uncertifiable while the stamp naming every option it ran under sat
    beside it, so the one comparison the ledger most needed came back as "nothing here can say".
    """
    import torch  # noqa: PLC0415 - only this test writes a stamped checkpoint

    control = _run(tmp_path, "ctl", CONTROL)
    control.with_suffix(".metrics.jsonl").unlink()
    torch.save({"options": {"scheme": "canonical"}, "epoch": 1, "holdout": 0.84}, control)
    arm = _run(tmp_path, "arm", ARM)
    _describe(arm, {"scheme": "extended"})

    comparison = RunComparison(control, [arm])

    assert comparison.recipe(comparison.control) == {"scheme": "canonical"}, "from the stamp, its only record"
    assert comparison.unmatched(comparison.arms[0]) == "scheme canonical→extended", "the axis, and only it"


def test_unmatched_says_so_when_a_run_recorded_no_recipe(tmp_path: Path):
    """Silence read as a match is the failure; every file predating the recipe is in that state."""
    report = RunComparison(_run(tmp_path, "ctl", CONTROL), [_run(tmp_path, "arm", ARM)]).report()
    assert "recipe unrecorded" in report


def test_reader_change(tmp_path: Path):
    """Which half moved decides which findings the arm may be judged on, so it is read, not declared.

    A swapped label table is the clearest reader change there is AND is waived by the confound check —
    the two questions read the same diff and must not share an answer.
    """
    control = _run(tmp_path, "ctl", CONTROL)
    reader, imaging, idle = (_run(tmp_path, name, ARM) for name in ("reader", "imaging", "idle"))
    _describe(control, {"labels": "v2.csv", "train_size": "392"})
    _describe(reader, {"labels": "v3.csv", "train_size": "392"})
    _describe(imaging, {"labels": "v2.csv", "train_size": "448"})
    _describe(idle, {"labels": "v2.csv", "train_size": "392"})

    comparison = RunComparison(control, [reader, imaging, idle])
    said = [comparison.reader_change(arm) for arm in comparison.arms]

    assert said[0].startswith("READER change (labels)"), "a label table is an axis here even if not a confound"
    assert said[1].startswith("MODEL change"), "an unnamed option acts through the pixels by default"
    assert "nothing" in said[2], "an arm identical to its control has no axis to be judged on"
    undescribed = RunComparison(control, [_run(tmp_path, "bare", ARM)])
    said_of_it = undescribed.reader_change(undescribed.arms[0])
    assert "kind unknown" in said_of_it, "a run with no recipe is not silently called a model change"


def _predictions(checkpoint: Path, shift: float, source: str = TRAINED) -> None:
    """The per-study record a run has kept since knee-mx17, on labels that are actually separable."""
    truth = (np.random.default_rng(2026).random((STUDIES, len(TARGETS))) < 0.5).astype(np.float32)
    # The SAME ruler for both runs — that is what makes them pairable — and its own noise per run, or
    # the two rank every study alike and the interval comes out exactly zero.
    noise = np.random.default_rng(int(shift * 1000)).random((STUDIES, len(TARGETS)))
    scores = truth * 0.3 + noise * 0.7
    HoldoutRecord(checkpoint).write(
        [f"s{index}" for index in range(STUDIES)],
        scores.astype(np.float64),
        truth,
        epoch=1,
        source=source,
    )


def test_paired(tmp_path: Path):
    """This pair's OWN interval, not the 2.2x ceiling borrowed from a null pair of seed twins."""
    control, arm = _run(tmp_path, "ctl", CONTROL), _run(tmp_path, "arm", ARM)
    _predictions(control, shift=0.0)
    _predictions(arm, shift=0.05)

    comparison = RunComparison(control, [arm])
    paired = comparison.paired(comparison.arms[0])

    assert paired is not None
    assert set(paired.widths) == set(TARGETS)
    assert all(width > 0 for width in paired.widths.values())
    assert paired.exact, "both runs kept the array their own numbers were read off"


def test_named():
    """A floor measured on a re-derivation says so: its own tolerance is wider than the width quoted."""
    widths = {"MCL": 0.01, "ACL": 0.02}

    assert "INDICATIVE" not in PairedWidths(widths, exact=True).named()
    assert "INDICATIVE" in PairedWidths(widths, exact=False).named()
    assert "0.0200" in PairedWidths(widths, exact=True).named(), "the widest label, not the first"


def test_paired_is_indicative_when_either_run_was_back_filled(tmp_path: Path):
    """The 65 runs already on disk can only be back-filled, and a back-fill is not the training array."""
    control, arm = _run(tmp_path, "ctl", CONTROL), _run(tmp_path, "arm", ARM)
    _predictions(control, shift=0.0)
    _predictions(arm, shift=0.05, source=DERIVED)

    comparison = RunComparison(control, [arm])
    paired = comparison.paired(comparison.arms[0])

    assert paired is not None
    assert not paired.exact
    assert "INDICATIVE" in comparison.floor(comparison.arms[0])[1]


def test_paired_is_none_when_a_run_kept_no_predictions(tmp_path: Path):
    """Sixty-five runs predate the record, and 'could not be measured' is not 'is wide'."""
    control, arm = _run(tmp_path, "ctl", CONTROL), _run(tmp_path, "arm", ARM)
    _predictions(control, shift=0.0)

    comparison = RunComparison(control, [arm])
    assert comparison.paired(comparison.arms[0]) is None


def test_scored(tmp_path: Path):
    """A control that predates the recorder is comparable again — its predictions state its scores.

    This is not hypothetical: `wsfold0_e6` is the control knee-bot's argument rests on, it wrote no
    metrics file, and the comparison refused it as "does not exist" while the arrays sat beside it.
    """
    control, arm = _run(tmp_path, "ctl", CONTROL), _run(tmp_path, "arm", ARM)
    control.with_suffix(".metrics.jsonl").unlink()
    _predictions(control, shift=0.0)

    comparison = RunComparison(control, [arm])
    resolved = comparison.scored(comparison.control)

    assert resolved is not None and resolved.reconstructed
    assert "RECONSTRUCTED" in comparison.report(), "and the report never passes it off as a recorded run"
    assert set(comparison.deltas()["arm.metrics"]) >= {"MCL", "Effusion", "MEAN"}


def test_floor(tmp_path: Path):
    """The report says WHICH bound each arm was marked against, because they differ several-fold."""
    control, arm = _run(tmp_path, "ctl", CONTROL), _run(tmp_path, "arm", ARM)
    comparison = RunComparison(control, [arm])

    assert "UNPAIRED" in comparison.floor(comparison.arms[0])[1]

    _predictions(control, shift=0.0)
    _predictions(arm, shift=0.05)

    intervals, named = RunComparison(control, [arm]).floor(comparison.arms[0])
    assert "paired floor" in named
    assert set(intervals) == set(TARGETS)


def test_resolved(tmp_path: Path):
    """A run named the way the ledger names it finds its checkpoint; a path is left exactly as given."""
    assert RunComparison.resolved(Path("centre_tissue_s2026"), tmp_path) == tmp_path / "centre_tissue_s2026.pt"
    assert RunComparison.resolved(Path("runs/x.pt"), tmp_path) == Path("runs/x.pt")
    assert RunComparison.resolved(Path("x.pt"), tmp_path) == Path("x.pt")


def test_resolved_accepts_the_name_the_tool_itself_prints(tmp_path: Path):
    """The `.metrics` form is a NAME, not a path — and it is the spelling the report and ledger use.

    The equivalence class a suffix rule gets wrong: `run.metrics` has a suffix and is still a bare name,
    so the old rule sent it to the current directory and reported the run missing. Every arm line this
    tool prints is in that spelling, which makes pasting one back the likeliest way to call it.
    """
    assert RunComparison.resolved(Path("lr4e-5_ws_s2026.metrics"), tmp_path) == tmp_path / "lr4e-5_ws_s2026.pt"
    assert RunComparison.resolved(Path("runs/x.metrics"), tmp_path) == Path("runs/x.metrics"), "a path stays a path"


def test_report_shows_each_labels_own_peak_epoch(tmp_path: Path):
    """A label can peak before the run's mean does; the trajectory reader makes that visible."""
    arm_rows: list[EpochRow] = [
        {"epoch": 0, "loss": 1.0, "auc": 0.80, "seconds": 10, "per_label": {"MCL": 0.70, "Effusion": 0.90}},
        {"epoch": 1, "loss": 0.9, "auc": 0.84, "seconds": 10, "per_label": {"MCL": 0.85, "Effusion": 0.87}},
    ]
    comparison = RunComparison(_run(tmp_path, "ctl", CONTROL), [_run(tmp_path, "arm", arm_rows)])
    report = comparison.report()

    assert "label peaks across epochs" in report
    assert "MCL 0.8500@e1" in report
    assert "Effusion 0.9000@e0" in report
