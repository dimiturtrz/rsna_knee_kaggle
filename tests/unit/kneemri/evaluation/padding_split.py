"""Splitting an already-written per-study readout by whether the study could fill the trained extent."""

import numpy as np
import pandas as pd
import pytest

from core.data.corpus.study_metadata import IDENTIFIER
from kneemri.evaluation.padding_split import FIELD_COLUMN, Coverage, PaddingSplit, SubsetToll
from kneemri.evaluation.scores import LabelScores

STUDIES = ["a", "b", "c", "d"]


def _metadata(widths: list[float | None]) -> pd.DataFrame:
    """One row per study, in the same order, with an absent width written as a real absence."""
    return pd.DataFrame({IDENTIFIER: STUDIES, FIELD_COLUMN: widths})


def _split(fills: list[bool], known: list[bool] | None = None) -> PaddingSplit:
    """A split over four studies whose truth alternates, so a subset AUC is defined on both sides."""
    truth = np.tile(np.array([[1.0], [0.0]], dtype=np.float32), (2, 12))
    measured = np.array(known if known is not None else [True] * len(fills), dtype=np.bool_)
    return PaddingSplit(LabelScores(), truth, Coverage(np.array(fills, dtype=np.bool_), measured))


def _passes() -> tuple[np.ndarray, np.ndarray]:
    """A kept pass that ranks the truth perfectly and a foreign pass that reverses two of the four."""
    kept = np.tile(np.array([[0.9], [0.1], [0.9], [0.1]], dtype=np.float64), (1, 12))
    foreign = np.tile(np.array([[0.9], [0.1], [0.1], [0.9]], dtype=np.float64), (1, 12))
    return kept, foreign


def test_toll() -> None:
    """The toll is what the SECOND decode cost — foreign minus kept, negative where it lost."""
    assert SubsetToll("fills", 4, 0.8800, 0.8500, (0.80, 0.90)).toll == pytest.approx(-0.0300)


def test_logged() -> None:
    """The n and the interval are printed WITH the toll, so a subset cannot be read as the pool."""
    line = SubsetToll("padded", 490, 0.8800, 0.8500, (0.8100, 0.8900)).logged()

    assert "n=490" in line
    assert "toll -0.0300" in line
    assert "[0.8100, 0.8900]" in line


def test_subset() -> None:
    """Both passes are re-scored over the SAME rows, so the difference is the toll and nothing else."""
    kept, foreign = _passes()
    mask = np.array([True, True, False, False], dtype=np.bool_)

    row = _split([True, True, False, False]).subset("fills", kept, foreign, mask)

    assert row.studies == 2
    assert row.kept == pytest.approx(1.0), "the kept pass ranks the two selected studies perfectly"
    assert row.foreign == pytest.approx(1.0), "and so does the foreign pass on THESE two"


def test_read() -> None:
    """The pooled row comes first — a run whose pooled toll does not reproduce may not be split."""
    kept, foreign = _passes()

    rows = _split([True, True, False, False]).read("m.pt", kept, foreign)

    assert [row.label for row in rows] == ["pooled", "fills", "padded"]
    assert rows[0].studies == 4
    assert rows[1].toll == pytest.approx(0.0), "the reversal is entirely in the padded half"
    assert rows[2].toll < 0.0


def test_members_of() -> None:
    """Only the kept keys name members, and the suffix is stripped without touching the study columns."""
    keys = ["studies", "truth", "m.pt::kept", "m.pt::foreign", "n.pt::kept"]

    assert _split([True]).members_of(keys) == ["m.pt", "n.pt"]


def test_padded() -> None:
    """Padded means MEASURED and short — an unanswered study is in neither half, not in this one."""
    coverage = Coverage(np.array([True, False, False], dtype=np.bool_), np.array([True, True, False], dtype=np.bool_))

    assert list(coverage.padded) == [False, True, False]


def test_measured() -> None:
    """A study whose width is unknown is in NEITHER subset — an unknown field is not a narrow one."""
    coverage = Coverage.measured(STUDIES, _metadata([180.0, 160.0, 120.0, None]), 160.0)

    assert list(coverage.fills) == [True, True, False, False]
    assert list(coverage.padded) == [False, False, True, False], "the unanswered study is not padded"
