"""Grading a report into label rows, on answers written by hand rather than generated.

Everything the model touches is behind `LocalReader.load`, so the prompt and the parse — the two halves
that decide what the table says — are testable without twenty gigabytes of weights.
"""

from pathlib import Path

import pandas as pd

from core.data.labels.report_reader import (
    ABSENT,
    LEVEL,
    PRESENT,
    UNSURE,
    WORDING,
    LocalReader,
    ReadingPrompt,
    ReportGrade,
    ReportReading,
)
from core.findings import CRITERIA, SPECIFICITY, TARGETS


def _answer(word: str) -> str:
    """A model answer grading every finding the same way."""
    body = ", ".join(f'"{target}": "{word}"' for target in TARGETS)
    return f"{{{body}}}"


def test_verdict_of():
    """The three verdicts, one per side of the assertion boundary plus the rung that asserts nothing."""
    assert ReportReading.verdict_of(LEVEL[ReportGrade.SEVERE]) == PRESENT
    assert ReportReading.verdict_of(LEVEL[ReportGrade.ABSENT]) == ABSENT
    assert ReportReading.verdict_of(LEVEL[ReportGrade.UNCLEAR]) == UNSURE


def test_criteria():
    """Every finding carries the host's own criterion, quoted rather than paraphrased."""
    criteria = ReadingPrompt().criteria()
    assert all(f"- {target}: " in criteria for target in TARGETS)
    assert CRITERIA["Medial Meniscus"] in criteria
    assert len(criteria.splitlines()) == len(TARGETS)


def test_system():
    """The instruction names the grades, forbids silence-as-normal, and states the host's threshold.

    The last two assertions are the point of the rubric-bearing prompt: without them the reader grades
    "intrasubstance degeneration without a definite tear" above absent, which is 448 measured cells.
    """
    system = ReadingPrompt().system()
    assert all(grade.value in system for grade in ReportGrade)
    assert "silence is not a normal result" in system
    assert SPECIFICITY in system
    assert CRITERIA["ACL"] in system
    assert all(WORDING[grade] in system for grade in ReportGrade)
    assert ReportGrade.__doc__ not in system


def test_user():
    """The report travels beside the exact finding names the answer has to be keyed by."""
    user = ReadingPrompt().user("Complete ACL tear.")
    assert "Complete ACL tear." in user
    assert all(target in user for target in TARGETS)


def test_messages():
    """Two turns, system first, so a chat template renders the instruction as an instruction."""
    messages = ReadingPrompt().messages("normal study")
    assert [turn["role"] for turn in messages] == ["system", "user"]
    assert "normal study" in messages[1]["content"]


def test_grade_of():
    """A spoken word becomes a grade; a word outside the vocabulary falls to the rung that claims nothing."""
    assert ReportReading.grade_of(" Present ") is ReportGrade.PRESENT
    assert ReportReading.grade_of("torn") is ReportGrade.UNCLEAR


def test_parse():
    """A clean answer grades all twelve."""
    reading = ReportReading.parse("a", _answer("absent"))
    assert reading.grades == dict.fromkeys(TARGETS, ReportGrade.ABSENT)


def test_parse_survives_a_preamble():
    """The reader is asked for bare JSON and sometimes writes a sentence first; the object still lands."""
    reading = ReportReading.parse("a", f"Here is the grading:\n{_answer('severe')}\nHope this helps.")
    assert reading.grades[TARGETS[0]] is ReportGrade.SEVERE


def test_parse_falls_back_rather_than_raising():
    """An unreadable answer leaves every cell exactly where the lexicon already had it, not in an exception."""
    reading = ReportReading.parse("a", "the report is in Greek and I cannot read it")
    assert reading.unread() == len(TARGETS)


def test_row():
    """A row carries value, confidence and verdict per finding, keyed as report_labels_v2.csv is."""
    row = ReportReading.parse("a", _answer("present")).row()
    assert row["StudyInstanceUID"] == "a"
    assert row[TARGETS[0]] == LEVEL[ReportGrade.PRESENT]
    assert row[f"{TARGETS[0]}__verdict"] == PRESENT
    assert row[f"{TARGETS[0]}__conf"] == 0.95


def test_annotated(tmp_path: Path):
    """Only the all-or-nothing annotated rows survive — a partly-annotated study is not one of them."""
    rows = [
        {"StudyInstanceUID": "full", "Report": "intact", **dict.fromkeys(TARGETS, 0.0)},
        {"StudyInstanceUID": "partial", "Report": "intact", **dict.fromkeys(TARGETS, 0.0)} | {TARGETS[0]: None},
        {"StudyInstanceUID": "none", "Report": "intact", **dict.fromkeys(TARGETS, None)},
    ]
    train = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(train, index=False)
    picked = LocalReader.annotated(train)
    assert list(picked["StudyInstanceUID"]) == ["full"]
    assert list(picked.columns) == ["StudyInstanceUID", "Report"]


def test_unread():
    """The number this whole module exists to lower: cells still on the rung that asserts nothing."""
    assert ReportReading.parse("a", _answer("unclear")).unread() == len(TARGETS)
    assert ReportReading.parse("a", _answer("possible")).unread() == 0
