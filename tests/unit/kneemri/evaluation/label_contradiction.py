from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.findings import STUDY, TARGETS
from kneemri.evaluation.label_contradiction import POSITIVE, SILENT, VERDICT_SUFFIX, LabelContradiction

NO = "NO"


def _census(
    spoke: list[list[bool]],
    read: list[list[bool]],
    predicted: list[list[float]],
    truth: list[list[float]],
) -> LabelContradiction:
    return LabelContradiction(
        spoke=np.array(spoke, dtype=bool),
        read_positive=np.array(read, dtype=bool),
        predicted=np.array(predicted, dtype=np.float64),
        truth=np.array(truth, dtype=np.float32),
    )


def _wide(read: bool, predicted: float, truth: float) -> LabelContradiction:
    """One study, twelve findings, every cell carrying the same three values."""
    width = len(TARGETS)
    return _census([[True] * width], [[read] * width], [[predicted] * width], [[truth] * width])


def test_labelled_positive() -> None:
    """The midpoint is the boundary, and a graded label exactly at it counts as positive."""
    census = _census([[True, True]], [[True, True]], [[0.9, 0.9]], [[0.5, 0.49]])
    assert census.labelled_positive().tolist() == [[True, False]]


def test_candidates() -> None:
    """A candidate needs the reader to have SPOKEN and both instruments on the SAME side of the label.

    The four columns are the four ways this can go: agreeing against the label (a candidate), agreeing
    WITH the label, disagreeing with each other (two cancelling disagreements, never a candidate), and
    the reader silent while the model contradicts.
    """
    census = _census(
        [[True, True, True, False]],
        [[True, False, True, True]],
        [[0.9, 0.1, 0.1, 0.9]],
        [[0.0, 0.0, 0.0, 0.0]],
    )
    assert census.candidates().tolist() == [[True, False, False, False]]


def test_per_finding() -> None:
    """One row per finding, carrying BOTH denominators — cells and that finding's positives."""
    table = _wide(read=False, predicted=0.0, truth=1.0).per_finding()
    assert len(table) == len(TARGETS)
    assert set(table.columns) >= {"finding", "candidates", "cells", "positives", "share_cells", "share_positives"}
    assert (table["candidates"] == 1).all(), "reader and model both call negative against a positive label"
    assert (table["share_positives"] == 1.0).all(), "one candidate over one positive"
    assert table["share_positives"].is_monotonic_decreasing, "worst-affected finding first"


def test_lines() -> None:
    """The census prints both denominators and the reader's own silence rate."""
    lines = _wide(read=False, predicted=0.0, truth=1.0).lines()
    assert f"{len(TARGETS)} candidate cells of {len(TARGETS)}" in lines[0]
    assert "100.0% of cells" in lines[1]
    assert len(lines) == len(TARGETS) + 2


def test_read_table(tmp_path: Path) -> None:
    """The reader's table is indexed by study, and a repeated study does not become two rows."""
    path = tmp_path / "table.csv"
    verdicts = {f"{target}{VERDICT_SUFFIX}": [POSITIVE, SILENT, NO] for target in TARGETS}
    pd.DataFrame({STUDY: ["a", "b", "a"], **verdicts}).to_csv(path, index=False)

    table = LabelContradiction.read_table(path)
    assert table.index.name == STUDY
    assert table.index.tolist() == ["a", "b"], "the second 'a' is dropped, not appended"


def test_align(tmp_path: Path) -> None:
    """Records that carry no holdout array are a harness fault, and must not read as a null census."""
    with pytest.raises(ValueError, match="carries a holdout array"):
        LabelContradiction.align(pd.DataFrame(index=pd.Index([], name=STUDY)), [tmp_path / "absent.holdout.npz"])


def _record(path: Path, studies: list[str], value: float) -> Path:
    width = len(TARGETS)
    np.savez(
        path,
        studies=np.array(studies),
        predictions=np.full((len(studies), width), value, np.float64),
        truth=np.zeros((len(studies), width), np.float32),
        epoch=np.array(1),
        source=np.array("test"),
    )
    return path


def test_align_unions_complementary_records(tmp_path: Path) -> None:
    """Records holding out DISJOINT folds cover the corpus together; an intersection would cover nothing."""
    verdicts = {f"{target}{VERDICT_SUFFIX}": [POSITIVE, POSITIVE] for target in TARGETS}
    table = pd.DataFrame({STUDY: ["a", "b"], **verdicts}).set_index(STUDY)
    census = LabelContradiction.align(
        table,
        [
            _record(tmp_path / "one.holdout.npz", ["a"], 0.9),
            _record(tmp_path / "two.holdout.npz", ["b"], 0.9),
        ],
    )
    assert census.predicted.shape == (2, len(TARGETS)), "both folds' studies are present, not their overlap"
    assert census.candidates().all(), "reader and model both call positive where the label says negative"
