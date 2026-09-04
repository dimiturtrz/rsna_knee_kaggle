"""Confident learning over a hand-built corpus, so every cell's right answer is known before the run.

The fixtures are arithmetic, never a data root: a planted mislabel is one whose held-out probability the
test WROTE, so "did the detector find it" is a question with an answer rather than a number to admire.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.findings import STUDY, TARGETS
from kneemri.evaluation.label_errors import (
    DISAGREEMENT,
    FALSE_POSITIVE,
    MISSED,
    VERDICT_SUFFIX,
    DetectedCells,
    DetectorGate,
    GateRow,
    GoldCensus,
    LabelErrorAudit,
    LabelErrorDetector,
    LabelErrorReport,
    ReaderTable,
)

STUDIES = 40
PLANTED = (3, 11, 27)
"""Rows whose verdict contradicts a confident held-out probability — the errors the detector must find."""


def _reader_table(verdicts: dict[str, list[str | None]]) -> pd.DataFrame:
    """A reader table carrying a verdict column for every finding named, and nothing else."""
    studies = [f"s{index}" for index in range(len(next(iter(verdicts.values()))))]
    columns = {f"{target}{VERDICT_SUFFIX}": column for target, column in verdicts.items()}
    return pd.DataFrame({STUDY: studies, **columns})


def _planted() -> tuple[ReaderTable, np.ndarray]:
    """One finding whose labels alternate, with three cells the probabilities confidently contradict.

    Every other finding is left unread, so a single column carries the whole question and the assertions
    are about cells rather than about an aggregate over twelve.
    """
    given = np.zeros((STUDIES, len(TARGETS)), dtype=bool)
    read = np.zeros_like(given)
    read[:, 0] = True
    given[:, 0] = np.arange(STUDIES) % 2 == 0
    predicted = np.full((STUDIES, len(TARGETS)), 0.5)
    predicted[:, 0] = np.where(given[:, 0], 0.9, 0.1)
    for row in PLANTED:
        predicted[row, 0] = 1.0 - predicted[row, 0]
    return ReaderTable([f"s{index}" for index in range(STUDIES)], given, read), predicted


def _census(errors: dict[int, str], cells: int = 6) -> GoldCensus:
    """A census over `cells` cells, marking the named positions as the named kind of disagreement."""
    false_positive = np.zeros(cells, dtype=bool)
    missed = np.zeros(cells, dtype=bool)
    for index, kind in errors.items():
        if kind == "YES":
            false_positive[index] = True
        else:
            missed[index] = True
    study = np.array([f"s{index}" for index in range(cells)])
    return GoldCensus(study, np.array([TARGETS[0]] * cells), false_positive, missed)


def _annotated(truths: dict[str, float]) -> pd.DataFrame:
    """The radiologist's rows in the columns `ReaderAgreement.cells` joins on, one finding graded."""
    rows = []
    for study, truth in truths.items():
        row: dict[str, object] = {STUDY: study, "Report": "effusion is present in the joint"}
        row |= {target: (truth if target == TARGETS[0] else np.nan) for target in TARGETS}
        rows.append(row)
    return pd.DataFrame(rows)


def test_reader_table_of():
    """The verdict matrix follows the CALLER's order, and an absent study is unread rather than negative."""
    table = _reader_table({TARGETS[0]: ["YES", "no", None], TARGETS[1]: ["UNK", "YES", "YES"]})

    reader = ReaderTable.of(table, ["s1", "s0", "absent"])

    assert reader.given[1, 0], "s0's YES survives the reindex to second position"
    assert not reader.given[0, 0], "a lowercase no is a read negative, not a positive"
    assert reader.read[0, 0], "s1 states a verdict on the first finding"
    assert not reader.read[2, 0], "a study the table does not carry is UNREAD"
    assert not reader.read[2, 1], "and unread on every finding, not just the one it was silent about"
    assert reader.read[1, 1] and not reader.given[1, 1], "s0's UNK is a stated verdict and a read NEGATIVE"
    assert not reader.read[:, 2].any(), "a finding with no verdict column is unread everywhere"


def test_label_error_detector_quality():
    """Self-confidence is the probability of the GIVEN label, and an unread cell scores NaN, never zero."""
    reader, predicted = _planted()

    quality = LabelErrorDetector(predicted, reader).quality()

    assert quality.shape == (STUDIES, len(TARGETS))
    assert np.isnan(quality[:, 1]).all(), "an unread finding is NaN so abstentions cannot outrank errors"
    assert quality[0, 0] == pytest.approx(0.9), "a YES the models agree with scores its own probability"
    assert quality[1, 0] == pytest.approx(0.9), "a NO they agree with scores the complement, not 0.1"
    for row in PLANTED:
        assert quality[row, 0] == pytest.approx(0.1), "a contradicted cell scores low, which is the ranking"
    assert quality[PLANTED[0], 0] < quality[0, 0]


def test_label_error_detector_flagged():
    """cleanlab sizes its own flag set, and it lands on the planted cells rather than on the agreements."""
    reader, predicted = _planted()

    flagged = LabelErrorDetector(predicted, reader).flagged()

    assert flagged.shape == (STUDIES, len(TARGETS))
    assert not flagged[:, 1].any(), "an unread finding cannot be flagged"
    assert flagged[list(PLANTED), 0].any(), "at least one planted mislabel is caught"
    assert flagged[:, 0].sum() <= STUDIES // 2, "the confident joint flags a minority, not the column"

    single = ReaderTable(reader.identifiers, np.ones_like(reader.given), reader.read)
    assert not LabelErrorDetector(predicted, single).flagged()[:, 0].any(), "one class cannot be a noise rate"


def test_gate_row_precision():
    """Caught over SPENT, which is what a budget buys, and NaN rather than a divide on an empty budget."""
    assert GateRow(50, 20, 140, 684).precision() == pytest.approx(0.4)
    assert np.isnan(GateRow(0, 0, 140, 684).precision())


def test_gate_row_recall():
    """Caught over the errors that EXIST, so the two ratios cannot be read for one another."""
    assert GateRow(50, 35, 140, 684).recall() == pytest.approx(0.25)
    assert np.isnan(GateRow(50, 0, 0, 684).recall())


def test_gate_row_baseline():
    """Random precision is the prevalence, and it does not move with the budget."""
    assert GateRow(50, 20, 140, 700).baseline() == pytest.approx(0.2)
    assert GateRow(200, 20, 140, 700).baseline() == pytest.approx(0.2), "prevalence, not a function of budget"
    assert np.isnan(GateRow(50, 0, 140, 0).baseline())


def test_gate_row_lift():
    """1.0 is chance, and a detector doing twice the prevalence reads 2.0."""
    assert GateRow(50, 20, 140, 700).lift() == pytest.approx(2.0)
    assert GateRow(50, 10, 140, 700).lift() == pytest.approx(1.0), "matching prevalence is chance"
    assert np.isnan(GateRow(50, 20, 0, 700).lift())


def test_gate_row_line():
    """The row carries its own denominators, so no number in it can be read without one."""
    line = GateRow(50, 20, 140, 700).line()

    assert "budget   50" in line
    assert "20/140" in line, "the catch is printed against the errors that exist"
    assert "random 0.200" in line and "lift 2.00x" in line


def test_detector_gate_ranked():
    """The worst-scoring cells are spent first, and the catch accumulates over the budgets asked for."""
    quality = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    gate = DetectorGate(quality, np.array([False, True, False, True, False, False]))

    rows = gate.ranked((1, 2, 6))

    assert [row.caught for row in rows] == [1, 2, 2], "both errors sit at the bottom of the ranking"
    assert [row.budget for row in rows] == [1, 2, 6]
    assert rows[0].errors == 2 and rows[0].scoreable == 6, "the denominators are the population, not the budget"
    assert gate.ranked((0, 7)) == [], "a budget larger than the population, or empty, is not a row"


def test_detector_gate_separation():
    """Perfect ordering reads 1.0, the reverse reads 0.0, and a tie-only ranking reads chance."""
    error = np.array([False, True, False, True])
    assert DetectorGate(np.array([0.9, 0.1, 0.8, 0.2]), error).separation() == pytest.approx(1.0)
    assert DetectorGate(np.array([0.1, 0.9, 0.2, 0.8]), error).separation() == pytest.approx(0.0)
    assert DetectorGate(np.full(4, 0.5), error).separation() == pytest.approx(0.5), "no ordering is chance"


def test_detector_gate_at_mask():
    """cleanlab's own set is priced at ITS size, which is the number a ranking cannot supply."""
    gate = DetectorGate(np.array([0.9, 0.1, 0.8, 0.2]), np.array([False, True, False, True]))

    row = gate.at_mask(np.array([False, True, True, False]))

    assert row.budget == 2, "the budget is the size the mask chose"
    assert row.caught == 1 and row.errors == 2 and row.scoreable == 4
    assert row.lift() == pytest.approx(1.0), "half of a two-cell flag on a half-error population is chance"


def test_gold_census_of():
    """A YES the radiologist denies is an over-call; anything else wrong is a miss."""
    cells = pd.DataFrame(
        {
            "study": ["s0", "s0", "s1"],
            "target": [TARGETS[0], TARGETS[1], TARGETS[0]],
            "verdict": ["YES", "UNK", "NO"],
            "truth": [0.0, 1.0, 1.0],
        }
    )

    census = GoldCensus.of(cells, cells.iloc[[0, 1, 2]])

    assert census.false_positive.tolist() == [True, False, False]
    assert census.missed.tolist() == [False, True, True], "UNK and NO are both under-calls, not over-calls"
    assert census.study.tolist() == ["s0", "s0", "s1"]
    assert census.target.tolist() == [TARGETS[0], TARGETS[1], TARGETS[0]]


def test_gold_census_disagreements():
    """Either direction, and an agreement is in neither."""
    census = _census({1: "YES", 4: "NO"})

    assert census.disagreements().tolist() == [False, True, False, False, True, False]
    assert int(census.disagreements().sum()) == int(census.false_positive.sum() + census.missed.sum())


def test_gold_census_counts():
    """The census counts ITSELF, under the same `kind` vocabulary the reader-agreement census uses."""
    counts = _census({1: "YES", 3: "YES", 4: "NO"}).counts()

    assert counts == {FALSE_POSITIVE: 2, MISSED: 1, DISAGREEMENT: 3}
    assert counts[DISAGREEMENT] == counts[FALSE_POSITIVE] + counts[MISSED], "no cell is wrong in both directions"
    assert _census({}).counts() == {FALSE_POSITIVE: 0, MISSED: 0, DISAGREEMENT: 0}


def test_gold_census_restricted():
    """All four arrays are cut TOGETHER, or the masks would describe cells the identifiers do not."""
    keep = np.array([True, True, False, False, True, False])

    cut = _census({1: "YES", 4: "NO"}).restricted(keep)

    assert cut.study.tolist() == ["s0", "s1", "s4"]
    assert cut.target.tolist() == [TARGETS[0]] * 3
    assert cut.false_positive.tolist() == [False, True, False]
    assert cut.missed.tolist() == [False, False, True], "the miss followed its own cell, not its old index"


def test_label_error_report_lines():
    """Three denominators on the coverage line, and one gate block per error class."""
    report = LabelErrorReport(
        corpus_studies=100,
        corpus_flagged=7,
        corpus_read=1000,
        annotated_studies=58,
        gold_studies=57,
        covered_studies=56,
        census=_census({1: "YES", 4: "NO"}),
        quality=np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.6]),
        flagged=np.array([False, True, False, False, True, False]),
    )

    lines = report.lines()

    assert "58 radiologist-annotated studies" in lines[1]
    assert "57 also carry a reader verdict" in lines[1] and "56 carry held-out predictions" in lines[1]
    assert "1 false positives, 1 misses, 2 disagreements of 6 cells" in lines[2]
    assert sum("separation AUC" in line for line in lines) == 3, "all disagreements, over-calls, misses"
    assert any("cleanlab's own mask" in line for line in lines)
    assert any("lift" in line for line in lines)


def test_label_error_audit_gated():
    """A gold cell the ruler does not cover is DROPPED and counted, never scored against a neighbour."""
    audit = LabelErrorAudit(
        table=_reader_table({TARGETS[0]: ["YES", "YES", "NO"]}),
        annotated=_annotated({"s0": 0.0, "s1": 1.0, "s2": 1.0}),
        published=Path("unused"),
    )
    identifiers = ["s2", "s0"]
    reader = ReaderTable.of(audit.table, identifiers)
    quality = np.zeros((len(identifiers), len(TARGETS)))
    quality[0, 0], quality[1, 0] = 0.8, 0.2

    report = audit.gated(identifiers, reader, quality, np.zeros_like(quality, dtype=bool))

    assert report.gold_studies == 3, "the census is defined over every annotated study with a verdict"
    assert report.covered_studies == 2, "and only the covered ones reach the gate"
    assert len(report.quality) == 2, "one cell per covered gold cell, s1 dropped whole"
    assert report.census.false_positive.tolist() == [True, False], "s0's YES against a 0.0 truth"
    assert report.census.missed.tolist() == [False, True], "s2's NO against a 1.0 truth"
    assert report.quality.tolist() == [0.2, 0.8], "the census keeps ITS order, and each cell its own score"
    assert report.corpus_studies == 2 and report.corpus_read == 2


def test_label_error_audit_report(monkeypatch: pytest.MonkeyPatch):
    """The ruler's seeds are AVERAGED before scoring, and the gate lands on the cells it covers."""
    identifiers = ["s0", "s1"]
    seeds = [np.full((2, len(TARGETS)), 0.2), np.full((2, len(TARGETS)), 0.8)]
    monkeypatch.setattr(
        "kneemri.evaluation.label_errors.PublishedRuler",
        lambda published: type("_Stub", (), {"per_seed": lambda self: (identifiers, seeds)})(),
    )
    audit = LabelErrorAudit(
        table=_reader_table({TARGETS[0]: ["YES", "NO"]}),
        annotated=_annotated({"s0": 0.0, "s1": 1.0}),
        published=Path("unused"),
    )

    report = audit.report()

    assert report.corpus_studies == 2 and report.covered_studies == 2
    assert report.quality.tolist() == [0.5, 0.5], "0.2 and 0.8 average to a cell neither seed reported"
    assert int(report.census.false_positive.sum()) == 1 and int(report.census.missed.sum()) == 1


def test_frame():
    """The detector's mask, written as the (study, finding) pairs `Supervision.pruned` joins on.

    Column NAMES are the contract — a frame with the right values under the wrong headers reads as zero
    flags at training time and every log line still looks healthy, which is the silent shape this pair
    of columns exists to make loud.
    """
    identifiers = ["a", "b", "c"]
    flagged = np.zeros((len(identifiers), len(TARGETS)), dtype=bool)
    flagged[2, 1] = True
    quality = np.zeros_like(flagged, dtype=np.float64)
    reader = ReaderTable(identifiers, flagged.copy(), flagged.copy())

    frame = DetectedCells(identifiers, reader, quality, flagged).frame()

    assert list(frame.columns) == ["StudyInstanceUID", "target"]
    assert len(frame) == 1
    assert frame.iloc[0]["StudyInstanceUID"] == "c"
    assert frame.iloc[0]["target"] == TARGETS[1]


def test_detected_cells_ranked():
    """The N worst-scoring READ cells — a budget, and never an unread cell whatever its score sorts as.

    `errors_dense` pruned the mask at full budget and cost −0.0158; the ranking is the same detector
    read at its strong end, so the one thing this must not do is let a NaN (unread) cell in by sorting.
    """
    identifiers = ["a", "b", "c"]
    quality = np.full((len(identifiers), len(TARGETS)), 0.9)
    quality[0, 0] = np.nan
    quality[1, 0] = 0.1
    quality[2, 0] = 0.2
    flagged = np.zeros_like(quality, dtype=bool)
    reader = ReaderTable(identifiers, flagged.copy(), flagged.copy())

    detected = DetectedCells(identifiers, reader, quality, flagged)
    chosen = detected.ranked(2)

    assert chosen.sum() == 2
    assert chosen[1, 0] and chosen[2, 0], "the two lowest scores, worst first"
    assert not chosen[0, 0], "an unread cell has no score to rank and may never be selected"
    assert list(detected.frame(chosen)["StudyInstanceUID"]) == ["b", "c"]


def test_detected(monkeypatch: pytest.MonkeyPatch):
    """The gate and the training flags are ONE estimate, so the published ruler is read once.

    A second read would also let the flags an arm trains against drift from the flags the gate passed,
    which is a disagreement neither number can show on its own.
    """
    identifiers = ["s0", "s1"]
    seeds = [np.full((2, len(TARGETS)), 0.2), np.full((2, len(TARGETS)), 0.8)]
    calls: list[int] = []

    def _counted(self: object) -> tuple[list[str], list[np.ndarray]]:
        calls.append(1)
        return identifiers, seeds

    monkeypatch.setattr(
        "kneemri.evaluation.label_errors.PublishedRuler",
        lambda published: type("_Stub", (), {"per_seed": _counted})(),
    )
    audit = LabelErrorAudit(
        table=_reader_table({TARGETS[0]: ["YES", "NO"]}),
        annotated=_annotated({"s0": 0.0, "s1": 1.0}),
        published=Path("unused"),
    )

    found = audit.detected()

    assert calls == [1], "one read, not one per reading"
    assert found.identifiers == identifiers
    assert found.flagged.shape == found.quality.shape
