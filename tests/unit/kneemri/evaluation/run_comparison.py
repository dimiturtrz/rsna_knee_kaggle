import json
from pathlib import Path

import numpy as np
import pytest

from kneemri.evaluation.run_comparison import PairedWidths, RunComparison
from kneemri.models.model import TARGETS
from kneemri.runs.holdout_record import DERIVED, TRAINED, HoldoutRecord
from kneemri.runs.run_metrics import EpochRow

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
