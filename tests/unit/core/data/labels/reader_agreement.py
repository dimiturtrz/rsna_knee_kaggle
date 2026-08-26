"""Reader-radiologist agreement per language, on constructed studies whose right answer is known by hand."""

import math

import pandas as pd
import pytest

from core.data.labels.reader_agreement import GRADED, HEDGED, ReaderAgreement
from core.findings import TARGETS

ENGLISH = "There is a tear of the medial meniscus with mild joint effusion and normal signal throughout."
SPANISH = "Técnica: RMN de la rodilla. Resultados: rotura parcial del menisco interno con derrame articular."
PLAIN = "There is a tear of the medial meniscus with joint effusion and normal signal throughout."
"""The same English report with its one severity word removed, so `graded` separates the two."""
HEDGED_REPORT = "There is possible tear of the medial meniscus; joint effusion cannot be excluded, signal normal."
"""English, unhedged vocabulary absent from both others, and carrying no severity word of its own."""


def _studies(reports: list[str], truths: list[list[float | None]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotated truth and reader verdicts for a handful of studies, one column per target."""
    identifiers = [f"s{index}" for index in range(len(reports))]
    columns: dict[str, object] = {"StudyInstanceUID": identifiers, "Report": reports}
    verdicts: dict[str, object] = {"StudyInstanceUID": identifiers}
    for position, target in enumerate(TARGETS):
        columns[target] = [row[position] if position < len(row) else None for row in truths]
        verdicts[f"{target}__verdict"] = ["YES" if position < len(row) else "UNK" for row in truths]
    return pd.DataFrame(columns), pd.DataFrame(verdicts)


def test_cells():
    """One row per annotated cell the reader also gave a verdict on, carrying the report's language.

    A cell the reader could not read is dropped rather than counted wrong: that would measure coverage
    while claiming to measure accuracy.
    """
    annotated, read = _studies([ENGLISH, SPANISH], [[1.0, 0.0], [1.0]])
    annotated.loc[0, TARGETS[1]] = None

    cells = ReaderAgreement().cells(annotated, read)

    assert len(cells) == 2, "one dropped for missing truth, one for a target the reader never verdicted"
    assert set(cells["target"]) <= set(TARGETS)
    assert set(cells["language"]) == {"en", "es"}
    assert list(cells["verdict"].unique()) == ["YES"]


def test_gap():
    """The English/non-English difference, and an interval that brackets it.

    Built so English agrees perfectly and Spanish never does, which is a gap no honest interval could
    report as absent — the guard against a measurement that always answers "no difference".
    """
    annotated, read = _studies([ENGLISH, ENGLISH, SPANISH, SPANISH], [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

    gap = ReaderAgreement().gap(ReaderAgreement().cells(annotated, read))

    assert gap["english"] == 1.0
    assert gap["other"] == 0.0
    assert gap["delta"] == 1.0
    assert gap["low"] <= gap["delta"] <= gap["high"], "the interval must contain the observed difference"


def test_report():
    """A line per language, then the gap line carrying both rates and the interval."""
    annotated, read = _studies([ENGLISH, SPANISH, SPANISH], [[1.0], [1.0], [0.0]])

    lines = ReaderAgreement().report(ReaderAgreement().cells(annotated, read))

    assert "3 annotated cells, 3 of them called positive" in lines[0]
    assert any(line.strip().startswith("es") and "50.0%" in line for line in lines)
    assert "over studies" in lines[-1]


def test_by_label():
    """Agreement per finding, each carrying its own count because the counts are small.

    The ruler is not equally good everywhere: a label the reader is right about half the time measures
    agreement with a coin as much as it measures the model, which is what decides whether optimising
    that label locally is safe at all.
    """
    annotated, read = _studies([ENGLISH, SPANISH, SPANISH], [[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])

    by_label = ReaderAgreement().by_label(ReaderAgreement().cells(annotated, read))

    assert by_label[TARGETS[0]] == (3, pytest.approx(2 / 3))
    assert by_label[TARGETS[1]] == (3, pytest.approx(1 / 3))


def _cells() -> pd.DataFrame:
    """Four cells: one right, one over-called, one missed while confident, one missed while unsure."""
    return pd.DataFrame(
        [
            ("s1", "en", "ACL", "YES", 1.0),
            ("s1", "en", "MCL", "YES", 0.0),
            ("s2", "en", "ACL", "NO", 1.0),
            ("s2", "en", "Synovitis", "UNK", 1.0),
        ],
        columns=["study", "language", "target", "verdict", "truth"],
    )


def test_split():
    """The difference between the cells a report-text flag fires on and the rest, with its interval.

    Built so the flagged studies never agree and the plain ones always do — a difference no honest
    interval can report as absent, which is the guard against a measurement that always answers "no".
    """
    annotated, read = _studies([ENGLISH, ENGLISH, PLAIN, PLAIN], [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    cells = ReaderAgreement().cells(annotated, read)

    split = ReaderAgreement().split(cells, GRADED)

    assert split["cells"] == 4, "two graded studies, two annotated cells each"
    assert split["fired"] == 0.0
    assert split["plain"] == 1.0
    assert split["delta"] == -1.0
    assert split["low"] <= split["delta"] <= split["high"], "the interval must contain the observed difference"


def test_split_needs_both_populations():
    """A flag that fires everywhere or nowhere answers NaN rather than raising.

    `hedged` fires on no language but English for want of vocabulary, so a homogeneous population is a
    real input here and not a malformed one; the bootstrap would otherwise draw from an empty array.
    """
    annotated, read = _studies([PLAIN, PLAIN], [[1.0], [0.0]])

    split = ReaderAgreement().split(ReaderAgreement().cells(annotated, read), GRADED)

    assert all(math.isnan(value) for value in split.values()), "every field NaN, none of them a number"


def test_signal_report():
    """One line per flag: how many calls it fires on, both rates, and the interval between them.

    HEDGED IS SCORED ON ENGLISH ONLY, so the Spanish study must not reach its line — outside English the
    flag is False for want of vocabulary rather than for want of hedging.
    """
    annotated, read = _studies([HEDGED_REPORT, PLAIN, SPANISH], [[1.0], [0.0], [0.0]])

    lines = ReaderAgreement().signal_report(ReaderAgreement().cells(annotated, read))

    assert len(lines) == 2
    assert lines[0].strip().startswith(GRADED) and "1 positive calls" in lines[0], "only the Spanish grades"
    assert lines[1].strip().startswith(HEDGED) and "1 positive calls" in lines[1]
    assert "agrees 100.0% against 0.0%" in lines[1], "the hedged study agrees, the plain English one does not"


def test_errors():
    """Only the cells the reader got wrong, labelled by DIRECTION rather than lumped together."""
    wrong = ReaderAgreement().errors(_cells())

    assert len(wrong) == 3, "the agreeing cell is not an error"
    assert set(wrong["kind"]) == {"false positive", "missed"}
    assert list(wrong[wrong["kind"] == "false positive"]["target"]) == ["MCL"]
    assert sorted(wrong[wrong["kind"] == "missed"]["target"]) == ["ACL", "Synovitis"]


def test_error_report():
    """Both directions, what the reader SAID on each, and findings ranked by how lopsided they are."""
    lines = ReaderAgreement().error_report(_cells())
    text = "\n".join(lines)

    assert "3 wrong of 4 cells over 2 studies" in text
    assert "missed: 2" in text and "UNK 1" in text and "NO 1" in text
    assert "false positive: 1" in text
    # MCL only over-calls (+1) and Synovitis only under-calls (-1), so they sit at opposite ends.
    assert text.index("MCL") < text.index("Synovitis"), "ranked by net over-calling, not by volume"


def _table(values: list[float], verdicts: list[str]) -> pd.DataFrame:
    """A label table: one row per study, every finding carrying the same value and verdict."""
    identifiers = [f"s{index}" for index in range(len(values))]
    columns: dict[str, object] = {"StudyInstanceUID": identifiers}
    for target in TARGETS:
        columns[target] = list(values)
        columns[f"{target}__verdict"] = list(verdicts)
    return pd.DataFrame(columns)


def test_blind():
    """A table whose verdicts are the default's while its numbers moved is refused, and says why."""
    default = _table([0.9, 0.2], ["YES", "NO"])
    blended = _table([0.7, 0.4], ["YES", "NO"])

    refusal = ReaderAgreement().blind(blended, default)

    assert "verdicts are" in refusal and "table_separation" in refusal, refusal
    assert f"{len(TARGETS)} findings" in refusal, "the refusal prices how much of the table moved"
    assert ReaderAgreement().blind(default, default) == "", "the default read against itself is readable"
    reread = _table([0.7, 0.4], ["NO", "NO"])
    assert ReaderAgreement().blind(reread, default) == "", "a table with its OWN verdicts is what this tool reads"


def _silent_cells() -> pd.DataFrame:
    """Six cells: three the reader spoke on, three it abstained on, only one of which hides a positive."""
    return pd.DataFrame(
        [
            ("s1", "en", "ACL", "YES", 1.0),
            ("s1", "en", "MCL", "YES", 0.0),
            ("s1", "en", "Effusion", "NO", 0.0),
            ("s2", "en", "Synovitis", "UNK", 1.0),
            ("s2", "en", "Fracture", "UNK", 0.0),
            ("s2", "en", "Baker's", "UNK", 0.0),
        ],
        columns=["study", "language", "target", "verdict", "truth"],
    )


def test_silence():
    """How many cells the reader abstained on, and how many of them the radiologist calls positive.

    The second number is the point: an UNK cell the radiologist also calls negative is already right at
    the prior, so `hidden` — not the abstention count — bounds what filling the silence can buy.
    """
    census = ReaderAgreement().silence(_silent_cells())

    assert census["cells"] == 3.0
    assert census["share"] == pytest.approx(0.5)
    assert census["hidden"] == 1.0, "only Synovitis hides a positive; two abstentions are already right"
    assert census["positive_silent"] == pytest.approx(1 / 3)
    assert census["positive_spoken"] == pytest.approx(1 / 3)


def test_silence_report():
    """The silence bound printed BESIDE the over-calling it competes with, and a per-finding split."""
    lines = ReaderAgreement().silence_report(_silent_cells())
    text = "\n".join(lines)

    assert "3 silent cells of 6 (50.0%)" in text
    assert "fixes at most 1 cells; over-calling costs 1" in text, "both failure modes on one line"
    assert text.index("Synovitis") < text.index("Fracture"), "ranked by hidden positives, not by silence"
