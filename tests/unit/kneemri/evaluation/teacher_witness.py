"""The witness is a residual correlation, so the tests state what the truth is supposed to remove."""

import logging
from pathlib import Path

import numpy as np
import pytest

from core import cli
from core.paths import DataRoot
from kneemri.evaluation import teacher_witness
from kneemri.evaluation.teacher_witness import FoldReadings, TeacherWitness

STUDIES = 64
"""Rows in each synthetic reading — enough that a correlation is not a two-point line."""

LABELS = 3
"""Columns, one per finding, since every number here is a mean over labels."""


def _truth() -> np.ndarray:
    return (np.arange(STUDIES * LABELS).reshape(STUDIES, LABELS) % 2).astype(np.float64)


def test_agreement() -> None:
    truth = _truth()
    noise = np.linspace(0.0, 1.0, STUDIES * LABELS).reshape(STUDIES, LABELS)
    assert TeacherWitness(truth).agreement(noise, noise) == 1.0


def test_agreement_reads_a_constant_column_as_zero() -> None:
    """A reading with no spread in a column says nothing there, which must read 0 rather than nan."""
    truth = _truth()
    varying = np.linspace(0.0, 1.0, STUDIES * LABELS).reshape(STUDIES, LABELS)
    flat = np.full((STUDIES, LABELS), 0.5)
    assert TeacherWitness(truth).agreement(flat, varying) == 0.0
    assert TeacherWitness(truth).agreement(varying, flat) == 0.0


def test_mimicry() -> None:
    truth = _truth()
    rng = np.random.default_rng(0)
    shared = rng.normal(size=(STUDIES, LABELS))
    apart = rng.normal(size=(STUDIES, LABELS))
    witness = TeacherWitness(truth)
    # Two readings that agree ONLY through the truth: their probabilities correlate, their residuals do not.
    assert witness.mimicry(truth + shared, truth + apart) < 0.5
    assert witness.mimicry(truth + shared, truth + shared) == 1.0


def test_lines() -> None:
    truth = _truth()
    reading = truth * 0.9 + 0.05
    lines = TeacherWitness(truth).lines({"student": reading}, {"leaky": reading})
    assert any("macro AUC" in line for line in lines)
    assert any("student" in line and "leaky" in line and "mimicry" in line for line in lines)


def test_aligned(tmp_path: Path) -> None:
    """A record is joined onto the card's study order by NAME, never by the order it was saved in."""
    record = tmp_path / "run.holdout.npz"
    np.savez(
        record,
        studies=np.array(["b", "a", "c"]),
        predictions=np.array([[0.2], [0.1], [0.3]]),
        truth=np.array([[1.0], [0.0], [1.0]]),
    )
    predictions, truth = FoldReadings.aligned(record, ["a", "c", "b"])
    assert predictions[:, 0].tolist() == [0.1, 0.3, 0.2]
    assert truth[:, 0].tolist() == [0.0, 1.0, 1.0]


def test_mean() -> None:
    """A teacher set's mean is taken over the members it named, and a missing one refuses."""
    read = [(Path("one.pt"), np.zeros((2, 1))), (Path("two.pt"), np.ones((2, 1)))]
    assert FoldReadings.mean(read, (Path("one.pt"), Path("two.pt")))[0, 0] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="was skipped"):
        FoldReadings.mean(read, (Path("one.pt"), Path("missing.pt")))


NAMES = ["a", "b", "c", "d"]
"""The fold the card reads, in the order it reads it."""


def _record(path: Path, scores: list[float]) -> Path:
    """A holdout record saved in REVERSED order, so only a join by name can line it up."""
    truth = np.array([[1.0], [0.0], [1.0], [0.0]])
    np.savez(path, studies=np.array(NAMES[::-1]), predictions=np.array(scores)[::-1, None], truth=truth[::-1])
    return path


def _witnessed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path, teachers: list[str]
) -> list[str]:
    models = tmp_path / "models"
    models.mkdir()
    for name in teachers:
        (models / name).write_bytes(b"")
    student = _record(tmp_path / "student.holdout.npz", [0.9, 0.2, 0.8, 0.1])
    control = _record(tmp_path / "control.holdout.npz", [0.6, 0.4, 0.5, 0.5])
    asked: list[list[Path]] = []

    def over(
        _self: FoldReadings, checkpoints: list[Path], studies: list[str]
    ) -> tuple[list[str], list[tuple[Path, np.ndarray]]]:
        asked.append(checkpoints)
        assert sorted(studies) == NAMES, "the card reads the student's own fold"
        return NAMES, [(path, np.full((len(NAMES), 1), 0.5)) for path in checkpoints]

    monkeypatch.setattr(cli.Command, "rooted", lambda self, arguments: DataRoot(tmp_path))
    monkeypatch.setattr(FoldReadings, "over", over)
    readings = ["--student", str(student), "--control", str(control)]
    monkeypatch.setattr("sys.argv", ["teacher_witness", *readings, "--models", str(models), "--device", "cpu"])
    caplog.set_level(logging.INFO, logger=teacher_witness.__name__)
    teacher_witness.main()
    assert len(asked) == 1
    return [record.getMessage() for record in caplog.records if record.name == teacher_witness.__name__]


def test_main_prints_one_row_per_reading_and_teacher_set(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    teachers = ["oof_fold1_s2026.pt", "oof_fold2_s2026.pt", "oof_clean_fold1_s2026.pt"]
    lines = _witnessed(monkeypatch, caplog, tmp_path, teachers)
    assert any(line.startswith("student") and "macro AUC 1.0000" in line for line in lines), "joined by name"
    pairs = [line for line in lines if " vs " in line]
    assert len(pairs) == 4, "two readings against two teacher sets"
    assert {line.split(" vs ")[1].split()[0] for line in pairs} == {"leaky", "clean"}


def test_main_refuses_an_empty_teacher_set(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="no clean teachers"):
        _witnessed(monkeypatch, caplog, tmp_path, ["oof_fold1_s2026.pt"])
