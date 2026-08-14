"""Reader-radiologist agreement per language, on constructed studies whose right answer is known by hand."""

import pandas as pd
import pytest

from kneemri.model import TARGETS
from kneemri.reader_agreement import ReaderAgreement

ENGLISH = "There is a tear of the medial meniscus with mild joint effusion and normal signal throughout."
SPANISH = "Técnica: RMN de la rodilla. Resultados: rotura parcial del menisco interno con derrame articular."


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
