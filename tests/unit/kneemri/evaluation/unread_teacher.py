from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data.labels.targets import SUSPECT_STUDY, SUSPECT_TARGET, UNREAD_VALUE
from core.findings import TARGETS
from kneemri.evaluation.unread_teacher import UNREAD_VERDICT, UnreadTeacher


def _labels(tmp_path: Path, verdicts: list[str]) -> Path:
    """A report table for as many studies as there are verdicts, all twelve findings sharing them."""
    frame: dict[str, list[object]] = {"StudyInstanceUID": [f"s{index}" for index in range(len(verdicts))]}
    for target in TARGETS:
        frame[target] = [0.136 if verdict == UNREAD_VERDICT else 0.958 for verdict in verdicts]
        frame[f"{target}__conf"] = [0.05 if verdict == UNREAD_VERDICT else 0.95 for verdict in verdicts]
        frame[f"{target}__verdict"] = list(verdicts)
    path = tmp_path / "report_labels.csv"
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


def test_band():
    """The half-width is the NEARER gap to a neighbouring measured level, so the tighter side caps it."""
    teacher = UnreadTeacher(Path("labels.csv"), Path("published"))

    assert teacher.band(0.28) == pytest.approx(0.20, abs=1e-9), (
        "an uncalibrated centre is measured against the RAW levels 0.08 and 0.68, never against the rates"
    )
    assert teacher.band(0.136) == pytest.approx(0.095, abs=1e-9), "0.041 below is nearer than 0.378 above"
    assert teacher.band(0.958) == pytest.approx(0.958 - 0.688, abs=1e-9), "at the top only one side exists"
    assert teacher.band(0.041) == pytest.approx(0.136 - 0.041, abs=1e-9)
    assert teacher.band(float(np.mean(np.full(4096, 0.28)))) == pytest.approx(0.20, abs=1e-9), (
        "a centre averaged out of one repeated constant is 0.28000000000000003, and the band must not "
        "read that 1e-17 as a neighbouring level and collapse to zero"
    )


def test_ordered():
    """Ranks are uniform on [0, 1] with ties averaged, which is what makes the band mean-preserving."""
    teacher = UnreadTeacher(Path("labels.csv"), Path("published"))

    assert teacher.ordered(np.array([0.9, 0.1, 0.5])).tolist() == pytest.approx([1.0, 0.0, 0.5])
    tied = teacher.ordered(np.array([0.2, 0.2, 0.9]))
    assert tied[0] == pytest.approx(tied[1]), "two identical predictions must not be split by sort order"
    assert teacher.ordered(np.array([0.4])).tolist() == [0.5], "a lone cell sits at the centre, not at an end"


def test_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Only UNK cells are written, centred on the constant they carry and inside the derived band."""
    labels = _labels(tmp_path, [UNREAD_VERDICT, UNREAD_VERDICT, UNREAD_VERDICT, "YES"])
    names = ["s0", "s1", "s2", "s3"]
    pool = np.tile(np.array([[0.1], [0.9], [0.5], [0.7]]), (1, len(TARGETS)))
    monkeypatch.setattr(
        "kneemri.evaluation.unread_teacher.PublishedRuler.per_seed",
        lambda self: (names, [pool]),
    )

    table = UnreadTeacher(labels, tmp_path / "published").table()

    assert set(table.columns) == {SUSPECT_STUDY, SUSPECT_TARGET, UNREAD_VALUE}
    assert len(table) == 3 * len(TARGETS), "three unread studies per finding, and the read one is left alone"
    assert set(table[SUSPECT_STUDY]) == {"s0", "s1", "s2"}
    one = table[table[SUSPECT_TARGET] == TARGETS[0]]
    assert one[UNREAD_VALUE].mean() == pytest.approx(0.136, abs=1e-6), "mean-preserving inside the block"
    assert one[UNREAD_VALUE].min() == pytest.approx(0.136 - 0.095, abs=1e-6)
    assert one[UNREAD_VALUE].max() == pytest.approx(0.136 + 0.095, abs=1e-6)
    assert one.set_index(SUSPECT_STUDY)[UNREAD_VALUE]["s1"] > one.set_index(SUSPECT_STUDY)[UNREAD_VALUE]["s0"], (
        "the pool's order is what the override carries; it is the only thing it adds"
    )


def test_table_without_a_verdict_column_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An older table carries no verdicts, and the answer to that is an empty table rather than a guess."""
    frame: dict[str, list[object]] = {"StudyInstanceUID": ["s0"]}
    for target in TARGETS:
        frame[target] = [0.28]
    labels = tmp_path / "old_labels.csv"
    pd.DataFrame(frame).to_csv(labels, index=False)
    monkeypatch.setattr(
        "kneemri.evaluation.unread_teacher.PublishedRuler.per_seed",
        lambda self: (["s0"], [np.zeros((1, len(TARGETS)))]),
    )

    table = UnreadTeacher(labels, tmp_path / "published").table()

    assert table.empty
    assert set(table.columns) == {SUSPECT_STUDY, SUSPECT_TARGET, UNREAD_VALUE}
