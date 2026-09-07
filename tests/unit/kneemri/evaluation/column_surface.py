"""The corpus surface as a ruler: the join it refuses, and the two readings it puts side by side."""

from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.column_surface import ColumnSurface, SurfaceRow

STUDIES = 40
RESAMPLES = 20
"""Enough draws for an interval to exist and be ordered; the published count is the module's default."""


def _identifiers(count: int = STUDIES) -> list[str]:
    return [f"study{index:03d}" for index in range(count)]


def _scores(shift: float = 0.0) -> np.ndarray:
    graded = np.linspace(0.0, 1.0, STUDIES)[:, None] + shift
    return np.repeat(graded, len(TARGETS), axis=1)


def _surface(tmp_path: Path, name: str, shift: float = 0.0) -> Path:
    path = tmp_path / name
    np.savez(path, identifiers=np.array(_identifiers()), scores=_scores(shift))
    return path.with_suffix(".npz") if path.suffix != ".npz" else path


def _labels(tmp_path: Path) -> Path:
    path = tmp_path / "report_labels_v2.csv"
    header = ",".join(["StudyInstanceUID", *TARGETS])
    positive = [index % 2 for index in range(STUDIES)]
    rows = [
        ",".join([identifier, *[f"{0.9 if flag else 0.1}"] * len(TARGETS)])
        for identifier, flag in zip(_identifiers(), positive, strict=True)
    ]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def _surfaced(tmp_path: Path) -> ColumnSurface:
    return ColumnSurface(_labels(tmp_path), resamples=RESAMPLES)


def test_half():
    """The half-width is what a frame comparison turns on, so it is a method and not a print format."""
    assert SurfaceRow("MCL", 4406, 272, 0.8451, 0.8214, 0.8688).half() == pytest.approx(0.0237, abs=1e-4)


def test_line():
    line = SurfaceRow("MCL", 4406, 272, 0.8451, 0.8214, 0.8688).line()

    assert "0.8451" in line
    assert "+/-0.0237" in line
    assert "n=4406 pos=272" in line


def test_read(tmp_path: Path):
    identifiers, scores = ColumnSurface.read(_surface(tmp_path, "surface.npz"))

    assert identifiers[:2] == ["study000", "study001"]
    assert scores.shape == (STUDIES, len(TARGETS))
    assert scores.dtype == np.float64


def test_read_refuses_a_file_without_identifiers(tmp_path: Path):
    """The refusal that keeps a positional join impossible — it produces a plausible AUC and no flag."""
    path = tmp_path / "bare.npz"
    np.savez(path, scores=_scores())

    with pytest.raises(ValueError, match="not a corpus surface"):
        ColumnSurface.read(path)


def test_onto(tmp_path: Path):
    """The rows come back in the ASKED order, not the surface's — the whole point of joining by name."""
    surface = ColumnSurface.read(_surface(tmp_path, "surface.npz"))

    projected = ColumnSurface.onto(["study002", "study000"], surface, "surface")

    assert projected[0, 0] == pytest.approx(surface[1][2, 0])
    assert projected[1, 0] == pytest.approx(surface[1][0, 0])


def test_onto_refuses_a_missing_study(tmp_path: Path):
    surface = ColumnSurface.read(_surface(tmp_path, "surface.npz"))

    with pytest.raises(ValueError, match="has no row for 1 of 2 studies"):
        ColumnSurface.onto(["study000", "absent"], surface, "surface")


def test_truth(tmp_path: Path):
    truth = _surfaced(tmp_path).truth(["study001", "study000"])

    assert truth.shape == (2, len(TARGETS))
    assert truth[0, 0] > ColumnSurface(Path()).scores.cut > truth[1, 0]


def test_rows(tmp_path: Path):
    """Twelve rows in the metric's column order, each carrying an interval that brackets its area."""
    surface = _surfaced(tmp_path)
    identifiers, scores = ColumnSurface.read(_surface(tmp_path, "surface.npz"))

    rows = surface.rows(scores, surface.truth(identifiers))

    assert [row.finding for row in rows] == list(TARGETS)
    assert rows[0].positives == STUDIES // 2
    assert rows[0].low <= rows[0].area <= rows[0].high


def test_delta(tmp_path: Path):
    """A surface against a monotone shift of itself: every column is ranked identically, so zero."""
    surface = _surfaced(tmp_path)
    identifiers, scores = ColumnSurface.read(_surface(tmp_path, "surface.npz"))
    shifted = ColumnSurface.onto(identifiers, ColumnSurface.read(_surface(tmp_path, "other.npz", shift=0.3)), "other")

    deltas = surface.delta(scores, shifted, surface.truth(identifiers))

    assert set(deltas) == set(TARGETS)
    point, low, high = deltas[TARGETS[0]]
    assert point == pytest.approx(0.0)
    assert low <= 0.0 <= high


def test_frame_studies(tmp_path: Path):
    path = tmp_path / "member.holdout.npz"
    np.savez(path, studies=np.array(_identifiers(3)), predictions=np.zeros((3, len(TARGETS))))

    assert ColumnSurface.frame_studies(path) == ["study000", "study001", "study002"]


def test_frame_studies_refuses_a_surface(tmp_path: Path):
    """A corpus surface spells its identifiers differently; reading one as a frame is the caller's slip."""
    with pytest.raises(ValueError, match="not a holdout record"):
        ColumnSurface.frame_studies(_surface(tmp_path, "surface.npz"))


def test_agreement():
    """A reordering of two columns has to cost rho — a metric that cannot see it measures nothing."""
    corpus = [
        SurfaceRow(name, 4406, 200, area, area - 0.01, area + 0.01)
        for name, area in zip("abc", [0.80, 0.85, 0.90], strict=True)
    ]
    kept = [SurfaceRow(row.finding, 896, 40, row.area, row.low, row.high) for row in corpus]
    swapped = [kept[1], kept[0], kept[2]]

    assert ColumnSurface.agreement(corpus, kept) == pytest.approx(1.0)
    assert ColumnSurface.agreement(corpus, swapped) < 1.0


def test_shift():
    """Both readings on one line, and the shift between them signed — the re-read this exists for."""
    corpus = [SurfaceRow("Fracture", 4406, 192, 0.8861, 0.8607, 0.9115)]
    frame = [SurfaceRow("Fracture", 896, 39, 0.8324, 0.7658, 0.8990)]

    lines = ColumnSurface.shift(corpus, frame)

    assert "finding" in lines[0]
    assert "+0.0537" in lines[1]
    assert "pos=39" in lines[1]
