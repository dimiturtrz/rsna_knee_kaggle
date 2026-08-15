"""The one premise knee-mxj rests on: whether a graded level means "worse" or only "surer"."""

import logging
from pathlib import Path

import numpy as np
import pytest

from kneemri.evaluation.holdout_record import HoldoutReading, HoldoutRecord
from kneemri.evaluation.ordinal_signal import CHANCE, OrdinalReading, OrdinalSignal, main
from kneemri.models.model import TARGETS

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
