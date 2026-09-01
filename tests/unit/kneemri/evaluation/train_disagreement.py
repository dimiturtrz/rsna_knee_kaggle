"""The mirror for the training-split disagreement instrument: every public method called, and asserted."""

import numpy as np

from core.findings import TARGETS
from kneemri.evaluation.train_disagreement import SPREAD_BANDS, SpreadRow, TrainDisagreement

STUDIES = 8


def _split(offsets: tuple[float, ...] = (0.0, 0.2, 0.4)) -> TrainDisagreement:
    """Three members whose disagreement rises with the study index."""
    base = np.tile(np.linspace(0.1, 0.9, STUDIES)[:, np.newaxis], (1, len(TARGETS)))
    tilt = np.tile(np.linspace(0.0, 1.0, STUDIES)[:, np.newaxis], (1, len(TARGETS)))
    return TrainDisagreement([np.clip(base + offset * tilt, 0.0, 1.0) for offset in offsets])


def test_members() -> None:
    assert _split().members() == 3


def test_spread() -> None:
    spread = _split().spread()
    assert spread.shape == (STUDIES,)
    assert spread[0] < spread[-1]


def test_weights() -> None:
    weights = _split().weights(0.5)
    assert weights.shape == (STUDIES,)
    assert (weights > 0.0).all()
    assert weights[0] < weights[-1]


def test_weights_at_zero_alpha_is_uniform() -> None:
    assert np.allclose(_split().weights(0.0), 1.0)


def test_banded() -> None:
    rows = _split().banded(0.5)
    assert len(rows) == len(SPREAD_BANDS)
    assert sum(row.studies for row in rows) == STUDIES


def test_line() -> None:
    line = SpreadRow(band="split (spread >= 0.15)", studies=12, mean_spread=0.2, mean_weight=1.5).line()
    assert "12" in line
    assert "0.2000" in line


def test_report() -> None:
    lines = _split().report(0.5)
    assert any("IN-SAMPLE" in line for line in lines)
    assert "by spread band" in lines
