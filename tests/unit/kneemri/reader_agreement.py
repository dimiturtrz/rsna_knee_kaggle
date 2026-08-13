"""Reader-radiologist agreement per language, on constructed studies whose right answer is known by hand."""

import pandas as pd

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
