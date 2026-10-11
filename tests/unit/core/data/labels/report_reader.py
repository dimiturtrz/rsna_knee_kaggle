"""Grading a report into label rows, on answers written by hand rather than generated.

Everything the model touches is behind `LocalReader.load`, so the prompt and the parse — the two halves
that decide what the table says — are testable without twenty gigabytes of weights.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data.labels.report_reader import (
    ABSENT,
    CONTUSION_CLARIFIED,
    CONTUSION_IMPACT_QUALIFIER,
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


def test_criteria_clarified():
    """A clarification lands on its own finding's line and leaves the other eleven byte-identical.

    The line count is the assertion that matters: a sentence appended to a criterion must not become a
    thirteenth line the reader can read as a thirteenth finding.
    """
    plain = ReadingPrompt().criteria().splitlines()
    clarified = ReadingPrompt(clarifications=CONTUSION_CLARIFIED).criteria().splitlines()
    assert len(clarified) == len(plain)
    changed = [new for old, new in zip(plain, clarified, strict=True) if old != new]
    assert len(changed) == 1
    assert changed[0].startswith("- Contusion: ")
    assert CONTUSION_IMPACT_QUALIFIER in changed[0]
    assert CRITERIA["Contusion"] in changed[0]


def test_system():
    """The instruction names the grades, forbids silence-as-normal, and states the host's threshold.

    The last two assertions are the point of the rubric-bearing prompt: without them the reader grades
    "intrasubstance degeneration without a definite tear" above absent, which is 502 measured cells whose
    negative class the radiologists' own annotations back.
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


def test_parse_falls_back_when_the_braces_hold_no_json():
    """An object-shaped answer that is not JSON costs its own cells, never the run."""
    reading = ReportReading.parse("a", "{MCL: present, ACL: absent}")
    assert reading.unread() == len(TARGETS)


class _Encoded(dict[str, object]):
    """What a tokenizer returns: a mapping that can be moved to the model's device."""

    def to(self, device: str) -> "_Encoded":
        self["device"] = device
        return self


class _Tokenizer:
    """Tokens are characters, so a decoded slice is readable text and the prompt's length is countable."""

    def __init__(self) -> None:
        self.template_flags: dict[str, object] = {}

    def apply_chat_template(self, messages: list[dict[str, str]], **flags: object) -> str:
        self.template_flags = flags
        return "".join(message["content"] for message in messages)

    def __call__(self, text: str, return_tensors: str) -> _Encoded:
        return _Encoded(input_ids=np.array([[ord(char) for char in text]]), tensors=return_tensors)

    def decode(self, tokens: np.ndarray, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return "".join(chr(int(token)) for token in tokens)


class _Model:
    """Echoes the prompt and appends a fixed answer, as a causal model's `generate` does."""

    device = "cpu"

    def generate(self, input_ids: np.ndarray, **kwargs: object) -> np.ndarray:
        assert kwargs["do_sample"] is False, "a grading must be reproducible"
        answer = np.array([[ord(char) for char in _answer("severe")]])
        return np.concatenate([input_ids, answer], axis=1)


def test_answer():
    """Only the GENERATED tokens come back, and the template is asked not to think first."""
    reader = LocalReader(prompt=ReadingPrompt())
    tokenizer = _Tokenizer()
    reader._loaded = (tokenizer, _Model())
    assert reader.answer("medial meniscus tear") == _answer("severe")
    assert tokenizer.template_flags["enable_thinking"] is False
    assert reader.read("a", "report").grades[TARGETS[0]] is ReportGrade.SEVERE


def test_table(monkeypatch: pytest.MonkeyPatch):
    """Every report becomes one row, in the order given, keyed by its own study."""
    answers = {"torn": _answer("severe"), "clean": _answer("absent"), "garbled": "no idea"}
    monkeypatch.setattr(LocalReader, "answer", lambda self, report: answers[report])
    reports = pd.DataFrame({"StudyInstanceUID": ["s1", "s2", "s3"], "Report": ["torn", "clean", "garbled"]})
    table = LocalReader().table(reports)
    assert list(table["StudyInstanceUID"]) == ["s1", "s2", "s3"]
    assert list(table[f"{TARGETS[0]}__verdict"]) == [PRESENT, ABSENT, UNSURE]


def test_table_of_nothing_is_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(LocalReader, "answer", lambda self, report: report)
    empty = pd.DataFrame({"StudyInstanceUID": [], "Report": []})
    assert LocalReader().table(empty).empty


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
