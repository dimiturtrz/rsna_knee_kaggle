"""What a corrected cell is told, asked of the table rather than of the arm that would train on it."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data.labels.targets import SUSPECT_STUDY, SUSPECT_TARGET, UNREAD_VALUE
from core.findings import TARGETS
from kneemri.evaluation.suspect_teacher import ALPHA, SuspectTeacher

LEVELS = [0.08, 0.28, 0.68, 0.94]


def _labels(tmp_path: Path, values: list[float]) -> Path:
    """A report table carrying one level per study, the same on every finding."""
    frame: dict[str, list[object]] = {"StudyInstanceUID": [f"s{index}" for index in range(len(values))]}
    for target in TARGETS:
        frame[target] = list(values)
    path = tmp_path / "report_labels.csv"
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


def _flags(tmp_path: Path, cells: list[tuple[str, str]]) -> Path:
    """The (study, target) table `label_errors --budget` writes."""
    path = tmp_path / "flags.csv"
    frame = pd.DataFrame(
        {SUSPECT_STUDY: [study for study, _ in cells], SUSPECT_TARGET: [target for _, target in cells]}
    )
    frame.to_csv(path, index=False)
    return path


def _teacher(tmp_path: Path, values: list[float], cells: list[tuple[str, str]], alpha: float = ALPHA) -> SuspectTeacher:
    return SuspectTeacher(_labels(tmp_path, values), tmp_path / "published", _flags(tmp_path, cells), alpha)


def test_expressed():
    """A pool score comes back as a value the READER could have written, not as its own probability."""
    teacher = SuspectTeacher(Path("labels.csv"), Path("published"), Path("flags.csv"))
    levels = np.array(LEVELS, dtype=np.float64)

    mapped = teacher.expressed(np.array([0.9, 0.1, 0.5, 0.7], dtype=np.float64), levels)
    assert set(np.round(mapped, 6)).issubset({round(level, 6) for level in LEVELS}), (
        "the whole point of the mapping is that the corrected value stays inside the label vocabulary"
    )
    assert mapped[0] == max(mapped), "the highest pool score takes the highest level"
    assert mapped[1] == min(mapped), "and the lowest takes the lowest"

    tied = teacher.expressed(np.array([0.2, 0.2, 0.9], dtype=np.float64), levels)
    assert tied[0] == pytest.approx(tied[1]), "two identical predictions must not be split by sort order"
    lone = teacher.expressed(np.array([0.4], dtype=np.float64), levels)
    assert lone[0] in LEVELS, "one cell carries no order, so a middle LEVEL is the honest answer"
    assert lone[0] not in (min(LEVELS), max(LEVELS)), "and never an end of the scale, which asserts the most"


def test_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Only the flagged cells are written, and each is the interpolation rather than the posterior."""
    teacher = _teacher(tmp_path, [0.94, 0.94, 0.08, 0.08], [("s0", TARGETS[0]), ("s2", TARGETS[1])])
    pool = np.tile(np.array([[0.1], [0.9], [0.5], [0.7]]), (1, len(TARGETS)))
    monkeypatch.setattr(
        "kneemri.evaluation.suspect_teacher.PublishedRuler.per_seed",
        lambda self: (["s0", "s1", "s2", "s3"], [pool]),
    )

    table = teacher.table()

    assert set(table.columns) == {SUSPECT_STUDY, SUSPECT_TARGET, UNREAD_VALUE}
    assert len(table) == 2, "two flagged cells in, two rows out; every other cell is untouched"
    flagged = table.set_index([SUSPECT_STUDY, SUSPECT_TARGET])[UNREAD_VALUE]
    corrected = float(flagged[("s0", TARGETS[0])])
    assert corrected < 0.94, "the pool ranks s0 lowest, so a 0.94 the reader asserted has to come DOWN"
    assert corrected > 0.08, (
        "and it must not land on the posterior alone: a wrong correction at full weight is a mislabel "
        "that now looks clean, which is the mechanism BoMD's 0.6 is against"
    )
    assert corrected == pytest.approx(ALPHA * 0.08 + (1.0 - ALPHA) * 0.94, abs=1e-6)


def test_table_at_full_posterior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """alpha = 1 is the ablation arm, and it has to be reachable rather than argued about."""
    teacher = _teacher(tmp_path, [0.94, 0.94, 0.08, 0.08], [("s0", TARGETS[0])], alpha=1.0)
    monkeypatch.setattr(
        "kneemri.evaluation.suspect_teacher.PublishedRuler.per_seed",
        lambda self: (["s0", "s1", "s2", "s3"], [np.tile(np.array([[0.1], [0.9], [0.5], [0.7]]), (1, len(TARGETS)))]),
    )

    assert float(teacher.table()[UNREAD_VALUE].iloc[0]) == pytest.approx(0.08, abs=1e-6)


def test_table_drops_a_flag_the_corpus_does_not_carry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A smoke run holds sixty studies and the detector flagged the whole corpus; that is not an error."""
    teacher = _teacher(tmp_path, [0.68, 0.68], [("s9", TARGETS[0])])
    monkeypatch.setattr(
        "kneemri.evaluation.suspect_teacher.PublishedRuler.per_seed",
        lambda self: (["s0", "s1"], [np.zeros((2, len(TARGETS)))]),
    )

    table = teacher.table()

    assert table.empty
    assert set(table.columns) == {SUSPECT_STUDY, SUSPECT_TARGET, UNREAD_VALUE}
