"""The label-versus-model split, on annotations written by hand so each side's answer is known."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.findings import TARGETS
from kneemri.evaluation.gold_rows import GoldRows
from kneemri.evaluation.supervision_gap import SupervisionGap

STUDIES = 8
SEED = 2026
RESAMPLES = 200
"""Enough resamples for a band to be stable at these separations, few enough to keep the unit fast."""


def _train_table(path: Path, *, annotate: int) -> Path:
    """A train.csv where the first `annotate` studies carry every finding and the rest carry none."""
    rows = []
    for index in range(STUDIES):
        row: dict[str, object] = {"StudyInstanceUID": f"s{index}"}
        for column, target in enumerate(TARGETS):
            row[target] = float((index + column) % 2) if index < annotate else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _corpus(tmp_path: Path, identifiers: list[str]) -> tuple[Path, Path]:
    """A labels table that reads the truth exactly, and one published member that inverts every odd finding.

    Chosen so each side of the split is unambiguous: the reference scores 1.0 everywhere, the model
    matches it on the even findings and scores 0.0 on the odd ones, so odd is model-bottlenecked and
    even is not. No ties anywhere, because a tie would put the answer at the mercy of a sort.
    """
    truth = np.array([[float((index + column) % 2) for column in range(len(TARGETS))] for index in range(STUDIES)])
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        [{"StudyInstanceUID": f"s{index}"} | dict(zip(TARGETS, truth[index], strict=True)) for index in range(STUDIES)]
    ).to_csv(labels, index=False)

    published = tmp_path / "published"
    published.mkdir()
    rows = [
        [value if column % 2 == 0 else 1.0 - value for column, value in enumerate(truth[int(name[1:])])]
        for name in identifiers
    ]
    torch.save({"oof": {"ids": identifiers, "pred": rows}, "config": {"seed": 2026}}, published / "f0_s2026.pt")
    return labels, published


def test_aligned(tmp_path: Path):
    """Only the studies with BOTH an annotation and a held-out prediction, on one order across all three."""
    train = _train_table(tmp_path / "train.csv", annotate=STUDIES - 2)
    labels, published = _corpus(tmp_path, ["s2", "s0", "s1"])

    values, predictions, annotations = SupervisionGap().aligned(labels, train, published)

    assert values.shape == predictions.shape == annotations.shape == (3, len(TARGETS))
    assert values[:, 0].tolist() == annotations[:, 0].tolist(), "the report table follows the annotation order"
    assert values[:, 1].tolist() != predictions[:, 1].tolist(), "and the odd finding is the inverted one"


def test_matched(tmp_path: Path):
    """Predictions handed in directly align the same way, and by NAME rather than by row order."""
    train = _train_table(tmp_path / "train.csv", annotate=STUDIES - 2)
    labels, _ = _corpus(tmp_path, ["s2", "s0", "s1"])
    names = ["s1", "s0"]
    combined = np.array([[float(column % 2) for column in range(len(TARGETS))] for _ in names], dtype=np.float32)

    values, predictions, annotations = SupervisionGap().matched(labels, train, names, combined)

    assert values.shape == predictions.shape == annotations.shape == (2, len(TARGETS))
    assert values.tolist() == annotations.tolist(), "the report table follows the annotation order"
    assert predictions[:, 1].tolist() == [1.0, 1.0], "and the handed-in rows were re-ordered by name, not copied"


def test_measured(tmp_path: Path):
    """The split, end to end from the three files: the findings the model inverts are the model's half."""
    train = _train_table(tmp_path / "train.csv", annotate=STUDIES)
    labels, published = _corpus(tmp_path, [f"s{index}" for index in range(STUDIES)])

    split = SupervisionGap().measured(labels, train, published)

    assert set(split.model_side) == set(TARGETS[1::2]), "the inverted findings, and only those"
    assert set(split.label_side) == set(TARGETS[0::2])


def test_annotated(tmp_path: Path):
    """The gold rows reach every caller through here unchanged; `GoldRows` owns what they mean."""
    train = _train_table(tmp_path / "train.csv", annotate=3)
    identifiers, truth = SupervisionGap().annotated(train)
    expected_names, expected = GoldRows().annotated(train)

    assert identifiers == expected_names
    assert np.array_equal(truth, expected), "a delegation that reshaped its answer would be a second reader"


def test_reported(tmp_path: Path):
    """Same delegation on the report side, so the two readers can never drift into different defaults."""
    path = tmp_path / "labels.csv"
    rows = [{"StudyInstanceUID": f"s{index}"} | dict.fromkeys(TARGETS, 0.94 - 0.1 * index) for index in (0, 1)]
    pd.DataFrame(rows).to_csv(path, index=False)
    order = ["s1", "s0", "absent"]

    assert np.array_equal(SupervisionGap().reported(path, order), GoldRows().reported(path, order))


def test_against():
    """One AUC per finding — a perfect predictor scores 1.0 and its negation scores 0.0."""
    truth = np.zeros((4, len(TARGETS)), dtype=np.float32)
    truth[2:, :] = 1.0
    gap = SupervisionGap()

    perfect = gap.against(truth.copy(), truth)
    assert perfect[TARGETS[0]] == 1.0

    inverted = gap.against(1.0 - truth, truth)
    assert inverted[TARGETS[0]] == 0.0, "the same measurement either way round, not an absolute value"


def test_separation():
    """The aggregate gap carries a band, and the band answers the question the point estimate cannot.

    Two equivalence classes, because only the two verdicts matter: labels that genuinely beat the
    model, where the paired interval must exclude zero and say so; and labels IDENTICAL to the model,
    where the difference is exactly zero on every resample and no reading may claim a weaker half.
    """
    generator = np.random.default_rng(SEED)
    truth = (generator.random((80, len(TARGETS))) > 0.5).astype(np.float32)
    values = truth + generator.normal(0.0, 0.25, truth.shape).astype(np.float32)
    predictions = truth + generator.normal(0.0, 1.4, truth.shape).astype(np.float32)
    gap = SupervisionGap()

    separated = gap.separation(values, predictions, truth, resamples=RESAMPLES)

    assert separated[0].startswith("labels - model: +"), "the noisier column is the weaker half"
    assert "over 80 studies" in separated[0], "the denominator is stated, never left to the reader"
    assert "EXCLUDES zero" in separated[1]
    assert "of 12 per-finding bands" in separated[2]

    tied = gap.separation(values, values.copy(), truth, resamples=RESAMPLES)

    assert tied[0].startswith("labels - model: +0.0000")
    assert "SPANS zero" in tied[1], "a column against itself may never be called the weaker half"

    swapped = gap.separation(predictions, values, truth, resamples=RESAMPLES)

    assert swapped[0].startswith("labels - model: -"), "the third class: the model beats its supervision"
    assert "EXCLUDES zero" in swapped[1] and "READER is the weaker half" in swapped[1], (
        "a wholly negative interval is as resolved as a wholly positive one"
    )


def test_verdict():
    """Three equivalence classes over one signed interval, because the reading has three answers.

    The middle class is the only one that was ever right by accident: until 2026-08-29 the negative
    interval fell through to it and a decisive model-wins result printed as silence.
    """
    gap = SupervisionGap()

    assert "model is the weaker half" in gap.verdict(0.02, 0.06)
    assert "READER is the weaker half" in gap.verdict(-0.0742, -0.0255), "the branch the real run needed"
    assert "SPANS zero" in gap.verdict(-0.01, 0.03)
    assert "SPANS zero" in gap.verdict(0.0, 0.03), "a boundary touching zero is not an exclusion"


def test_bottlenecked():
    """The split an arm is judged on: a reader change is only allowed the findings a reader can move.

    A finding whose gap cannot be computed must land on a side rather than vanish — a finding in
    neither group is a finding no arm is ever evaluated on, which is the silent kind of exclusion.
    """
    reference = {TARGETS[0]: 0.80, TARGETS[1]: 0.95, TARGETS[2]: 0.70, TARGETS[3]: float("nan")}
    achieved = {TARGETS[0]: 0.90, TARGETS[1]: 0.60, TARGETS[2]: 0.72, TARGETS[3]: 0.75}

    split = SupervisionGap().bottlenecked(reference, achieved)

    assert split.model_side == (TARGETS[1],), "only where the report beats the model"
    assert split.label_side[:2] == (TARGETS[0], TARGETS[2]), "biggest model-over-label gain first"
    assert set(split.label_side) | set(split.model_side) == set(reference), "every finding lands"
    assert TARGETS[3] in split.label_side, "an incomparable gap is not silently dropped"


def test_report():
    """Ordered by the GAP, with the dividing line drawn where the model stops beating its labels."""
    reference = {TARGETS[0]: 0.80, TARGETS[1]: 0.95, TARGETS[2]: 0.70}
    achieved = {TARGETS[0]: 0.90, TARGETS[1]: 0.60, TARGETS[2]: 0.72}

    lines = SupervisionGap().report(reference, achieved)
    text = "\n".join(lines)

    assert "aggregate:" in lines[0]
    assert text.index(TARGETS[0]) < text.index(TARGETS[2]), "biggest model-over-label gain first"
    assert text.index(TARGETS[2]) < text.index("--- above"), "a positive gap sits above the line"
    assert text.index("--- above") < text.index(TARGETS[1]), "and the model-bottlenecked finding below it"
    assert "+0.1000" in text


def test_thresholds():
    """Where a graded label becomes positive, priced by base rate and precision against the truth."""
    truth = np.zeros((4, len(TARGETS)), dtype=np.float32)
    truth[2:, :] = 1.0
    # The lower band is the reader's "mild" level: called present at 0.5, excluded at 0.75, and WRONG.
    values = np.full((4, len(TARGETS)), 0.68, dtype=np.float32)
    values[2:, :] = 0.94

    lines = SupervisionGap().thresholds(values, truth, cuts=(0.5, 0.75))

    assert "the radiologist calls 50.0% positive" in lines[0]
    assert "100.0%" in lines[2], "at 0.5 every cell is called positive"
    assert "50.0%" in lines[3], "at 0.75 only the top band is, which here is exactly right"
    assert lines[3].strip().endswith("100.0%"), "and it recovers every true positive"
