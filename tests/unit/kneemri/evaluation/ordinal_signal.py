"""The one premise knee-mxj rests on: whether a graded level means "worse" or only "surer"."""

import logging
from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.ordinal_signal import (
    CHANCE,
    OrdinalReading,
    OrdinalScan,
    OrdinalSignal,
    OrdinalSummary,
    main,
)
from kneemri.runs.holdout_record import HoldoutReading, HoldoutRecord

LEVELS = [0.68, 0.82, 0.94]


def _reading(column_truth: list[float], column_predictions: list[float]) -> HoldoutReading:
    truth = np.full((len(column_truth), len(TARGETS)), 0.5, dtype=np.float32)
    predictions = np.zeros((len(column_truth), len(TARGETS)), dtype=np.float64)
    truth[:, 0] = column_truth
    predictions[:, 0] = column_predictions
    return HoldoutReading(
        studies=[f"s{position}" for position in range(len(column_truth))],
        predictions=predictions,
        truth=truth,
        epoch=1,
    )


def test_concordance():
    """A ranking that agrees with the levels scores 1.0, one that inverts them 0.0, chance in between."""
    agreeing = OrdinalSignal.concordance(np.array([0.1, 0.5, 0.9]), np.array(LEVELS, dtype=np.float32))

    assert agreeing is not None
    assert agreeing[0] == pytest.approx(1.0)
    assert agreeing[1] == 3, "three studies at three levels make three cross-level pairs"
    assert agreeing[2] == pytest.approx(LEVELS, abs=1e-6), "the levels are float32 on disk, widened here"

    inverted = OrdinalSignal.concordance(np.array([0.9, 0.5, 0.1]), np.array(LEVELS, dtype=np.float32))
    assert inverted is not None
    assert inverted[0] == pytest.approx(0.0)

    flat = OrdinalSignal.concordance(np.array([0.4, 0.4, 0.4]), np.array(LEVELS, dtype=np.float32))
    assert flat is not None
    assert flat[0] == pytest.approx(CHANCE), "a constant predictor knows nothing, and midranks must say so"


def test_concordance_ignores_annotated_and_unmentioned_studies():
    """0.0/1.0 are a radiologist's verdict and 0.5 is silence; only the reader's grades are severity.

    `Targets.load` overwrites the derived value with a hard annotation where one exists, so pooling
    those in would measure the present-vs-absent signal this test exists to hold out — and it would do
    it in the direction that flatters the hypothesis, since the binary signal is the one the model has.
    """
    truth = np.array([0.0, 0.5, 1.0, 0.68, 0.94], dtype=np.float32)
    measured = OrdinalSignal.concordance(np.array([0.9, 0.9, 0.9, 0.8, 0.1]), truth)

    assert measured is not None
    assert measured[1] == 1, "one graded pair, not ten"
    assert measured[0] == pytest.approx(0.0), "and it is the inverted one, undiluted by the annotations"


def test_concordance_is_undefined_below_two_levels():
    """A label whose graded studies all sit at one level cannot answer, and must not answer chance."""
    assert OrdinalSignal.concordance(np.array([0.2, 0.8]), np.array([0.68, 0.68], dtype=np.float32)) is None
    assert OrdinalSignal.concordance(np.array([0.2, 0.8]), np.array([0.0, 1.0], dtype=np.float32)) is None


def test_per_label():
    """Each label answers for its own graded studies, and a label that cannot answer is absent."""
    readings = OrdinalSignal(_reading(LEVELS, [0.1, 0.5, 0.9])).per_label()

    assert [reading.label for reading in readings] == [TARGETS[0]], "the other eleven columns are all 0.5"
    assert readings[0].concordance == pytest.approx(1.0)
    assert readings[0].pairs == 3


def test_line():
    """The denominator travels with the number: pairs are not the holdout's size, and a rare finding has few."""
    printed = OrdinalReading(TARGETS[0], 0.61, 42, LEVELS).line()

    assert "0.610" in printed
    assert "42" in printed
    assert "0.68" in printed


def test_pooled():
    """The two means come apart exactly when the evidence is unevenly spread, which is the normal case."""
    readings = [OrdinalReading("common", 0.9, 900, LEVELS), OrdinalReading("rare", 0.5, 100, LEVELS)]

    summary = OrdinalSignal.pooled(readings)

    assert summary is not None
    assert summary.label_mean == pytest.approx(0.7), "each label counts once"
    assert summary.pair_weighted == pytest.approx(0.86), "each PAIR counts once, so the common label carries it"
    assert (summary.labels, summary.pairs) == (2, 1000)

    assert OrdinalSignal.pooled([]) is None, "no label could answer, and a mean of nothing is not chance"


def test_ordinal_summary_lines():
    """Neither mean is printed as THE mean: each row names the denominator it was taken over."""
    printed = OrdinalSummary(0.7, 0.86, 2, 1000).lines()

    assert len(printed) == 2, "one number called the mean is the reading error this exists to prevent"
    assert "0.700" in printed[0] and "labels" in printed[0]
    assert "0.860" in printed[1] and "pairs" in printed[1]


def test_ordinal_signal_detail(tmp_path: Path):
    """A member with no record says so as a line, rather than raising — 65 on disk predate the record."""
    checkpoint = tmp_path / "member.pt"
    assert "no holdout record" in OrdinalSignal.detail(checkpoint)[0]

    reading = _reading(LEVELS, [0.1, 0.5, 0.9])
    HoldoutRecord(checkpoint).write(reading.studies, reading.predictions, reading.truth, reading.epoch)
    printed = OrdinalSignal.detail(checkpoint)
    assert printed[0].startswith("member.pt")
    assert any(TARGETS[0] in line for line in printed)
    assert sum("MEAN" in line for line in printed) == 2, "both means, never one called THE mean"


def test_ordinal_scan_of(tmp_path: Path):
    """Members read off disk, and one with no record skipped rather than seated at chance."""
    agreeing, inverted = tmp_path / "agreeing.pt", tmp_path / "inverted.pt"
    for checkpoint, predictions in ((agreeing, [0.1, 0.5, 0.9]), (inverted, [0.9, 0.5, 0.1])):
        reading = _reading(LEVELS, predictions)
        HoldoutRecord(checkpoint).write(reading.studies, reading.predictions, reading.truth, reading.epoch)

    scan = OrdinalScan.of([agreeing, inverted, tmp_path / "never_trained.pt"])

    assert [name for name, _ in scan.scanned] == ["agreeing", "inverted"], "a recordless member is not chance"
    assert scan.scanned[0][1].pair_weighted == pytest.approx(1.0)


def test_ordinal_scan_ranked():
    """Best first by PAIRS, which is the weighting a seat displacement is decided under."""
    weak = OrdinalSummary(0.9, 0.60, 12, 1000)
    strong = OrdinalSummary(0.5, 0.75, 12, 1000)

    assert [name for name, _ in OrdinalScan([("weak", weak), ("strong", strong)]).ranked()] == ["strong", "weak"]


def test_ordinal_scan_lines():
    """The seated roster is marked, because outranking an unseated member is not a submission."""
    ranked = [("candidate", OrdinalSummary(0.5, 0.75, 12, 1000)), ("seated", OrdinalSummary(0.9, 0.6, 12, 9))]
    scan = OrdinalScan(ranked)

    printed = scan.lines(seated=("seated",))

    assert printed[1].startswith("candidate") and "SEAT" not in printed[1]
    assert "SEAT" in printed[2], "the row that a displacement would have to beat is the marked one"
    assert "0.900" in printed[2], "the label mean travels along, so a one-weighting-only ranking is visible"


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    """From a checkpoint path to a per-label table, with a run that has no record saying so plainly."""
    checkpoint = tmp_path / "member.pt"
    monkeypatch.setattr("sys.argv", ["ordinal_signal", "--checkpoint", str(checkpoint)])

    with caplog.at_level(logging.INFO):
        main()
    assert "no holdout record" in caplog.text, "65 members on disk predate the record, and must not raise"

    reading = _reading(LEVELS, [0.1, 0.5, 0.9])
    HoldoutRecord(checkpoint).write(reading.studies, reading.predictions, reading.truth, reading.epoch)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        main()

    assert TARGETS[0] in caplog.text
    assert "1.000" in caplog.text
