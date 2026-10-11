"""The middle-rung re-spread: which cells it may touch, how far it moves them, and what it may not cross."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import cli
from core.data.labels import band_resolution
from core.data.labels.band_resolution import CALIBRATED, MIDDLE, RAW, BandResolution
from core.data.labels.targets import MEASURED_AGREEMENT
from core.findings import TARGETS
from core.paths import DataRoot

STUDIES = 8
"""Rows in the fixture table — enough that a within-column percentile has something to spread."""

SHARE = 0.5
"""The registered share: a band cell may move half the way to a neighbouring rung and no further."""

SEED = 11
"""Fixed, because a unit test may not draw from an unseeded generator."""

BAND = MEASURED_AGREEMENT[MIDDLE][RAW]
LOWER = MEASURED_AGREEMENT[MIDDLE - 1][RAW]
UPPER = MEASURED_AGREEMENT[MIDDLE + 1][RAW]
"""The raw scale's middle rung and its two neighbours — the scale `report_labels_v2.csv` is written in."""


def _table() -> pd.DataFrame:
    """Every finding's first four cells on the rung and the rest off it, at the neighbouring values."""
    frame = pd.DataFrame({"StudyInstanceUID": [f"study_{index:05d}" for index in range(STUDIES)]})
    for target in TARGETS:
        column = np.full(STUDIES, UPPER)
        column[:4] = BAND
        column[4] = LOWER
        frame[target] = column
    return frame


def _scores() -> np.ndarray:
    generator = np.random.default_rng(SEED)
    return generator.random((STUDIES, len(TARGETS)))


def test_scales() -> None:
    """The two scales select the same cells at different values, which is why the switch exists."""
    assert MEASURED_AGREEMENT[MIDDLE][CALIBRATED] != BAND


def test_rung() -> None:
    """A different rung brackets against different neighbours, and selects none of the default's cells."""
    resolution = BandResolution(_table(), SHARE, RAW, MIDDLE + 1)
    assert resolution.band == pytest.approx(UPPER)
    assert resolution.lower == pytest.approx(BAND)
    assert int(resolution.banded(TARGETS[0]).sum()) == STUDIES - 5


def test_banded() -> None:
    """Only the cells sitting on the rung are selectable, and a CSV round trip must not lose them."""
    resolution = BandResolution(_table(), SHARE)
    band = resolution.banded(TARGETS[0])
    assert int(band.sum()) == 4
    assert bool(band.iloc[0])
    assert not bool(band.iloc[4])


def test_percentiles() -> None:
    """The order maps onto [0, 1] within the cells handed in, and a lone cell stays on the rung."""
    resolution = BandResolution(_table(), SHARE)
    spread = resolution.percentiles(np.array([0.9, 0.1, 0.5]))
    assert spread.min() == 0.0
    assert spread.max() == pytest.approx(1.0)
    assert list(np.argsort(spread)) == [1, 2, 0]
    assert resolution.percentiles(np.array([0.4]))[0] == pytest.approx(0.5)


def test_reach() -> None:
    """The bracket is the NEARER neighbour's distance both ways, so the band's mean does not move."""
    resolution = BandResolution(_table(), SHARE)
    assert resolution.reach() == pytest.approx(min(BAND - LOWER, UPPER - BAND))


def test_spread() -> None:
    """At share 0.5 the extremes move half the reach each way, symmetrically about the rung."""
    resolution = BandResolution(_table(), SHARE)
    moved = resolution.spread(np.array([0.0, 0.5, 1.0]))
    reach = SHARE * min(BAND - LOWER, UPPER - BAND)
    assert moved[0] == pytest.approx(BAND - reach)
    assert moved[1] == pytest.approx(BAND)
    assert moved[2] == pytest.approx(BAND + reach)
    assert moved.mean() == pytest.approx(BAND)
    assert moved.min() > LOWER
    assert moved.max() < UPPER


def test_resolved() -> None:
    """Band cells move and carry the model's order; every other cell is left exactly as it was."""
    table = _table()
    scores = _scores()
    written = BandResolution(table, SHARE).resolved(scores)
    for index, target in enumerate(TARGETS):
        assert list(np.argsort(written[target][:4])) == list(np.argsort(scores[:4, index]))
        assert written[target].iloc[4] == pytest.approx(LOWER)
        assert written[target].iloc[5] == pytest.approx(UPPER)


def test_resolved_at_share_zero() -> None:
    """Share 0.0 is the identity — the dial prices the whole treatment, so its floor must be a no-op."""
    table = _table()
    written = BandResolution(table, 0.0).resolved(_scores())
    for target in TARGETS:
        assert written[target].to_numpy() == pytest.approx(table[target].to_numpy())


def test_census() -> None:
    body = "\n".join(BandResolution(_table(), SHARE).census(_scores()))
    assert "of the table" in body
    assert TARGETS[0] in body


def test_resolved_leaves_a_column_with_no_band_cells_untouched() -> None:
    """A finding the reader never put on the rung has nothing to order, so its column is copied as it is."""
    table = _table()
    table[TARGETS[0]] = UPPER
    written = BandResolution(table, SHARE).resolved(_scores())
    assert written[TARGETS[0]].to_numpy() == pytest.approx(np.full(STUDIES, UPPER))
    assert not np.allclose(written[TARGETS[1]].to_numpy()[:4], BAND), "the other columns still move"


def _resolve(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, folds: dict[str, list[str]]) -> Path:
    """Write the table and the given out-of-fold records, run the CLI, and return where it wrote."""
    table = tmp_path / "labels.csv"
    _table().to_csv(table, index=False)
    names = list(_table()["StudyInstanceUID"])
    scores = _scores()
    for fold, studies in folds.items():
        rows = [names.index(study) for study in studies]
        np.savez(tmp_path / f"{fold}.holdout.npz", studies=np.array(studies), predictions=scores[rows])
    out = tmp_path / "out" / "resolved.csv"
    monkeypatch.setattr(cli.Command, "rooted", lambda self, arguments: DataRoot(tmp_path))
    stitch = ["--stitch", "oof_fold*.holdout.npz", "--share", str(SHARE), "--out", str(out)]
    monkeypatch.setattr("sys.argv", ["band_resolution", "--table", str(table), "--models", str(tmp_path), *stitch])
    band_resolution.main()
    return out


def test_main_writes_the_table_re_spread_by_the_stitched_folds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Two folds partitioning the table stitch into one ordering, and the written table carries it."""
    names = list(_table()["StudyInstanceUID"])
    out = _resolve(monkeypatch, tmp_path, {"oof_fold0": names[::2], "oof_fold1": names[1::2]})
    written = pd.read_csv(out)
    expected = BandResolution(_table(), SHARE).resolved(_scores())
    for target in TARGETS:
        assert written[target].to_numpy() == pytest.approx(expected[target].to_numpy(), abs=1e-9)


def test_main_refuses_when_no_record_matches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no fold records"):
        _resolve(monkeypatch, tmp_path, {})


def test_main_refuses_folds_that_overlap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A study read by two folds would carry two orderings; the stitch must be a partition."""
    names = list(_table()["StudyInstanceUID"])
    with pytest.raises(ValueError, match="not a partition"):
        _resolve(monkeypatch, tmp_path, {"oof_fold0": names, "oof_fold1": names[:1]})


def test_main_refuses_a_table_row_no_fold_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A row without an out-of-fold score would be ordered by nothing, so the whole write refuses."""
    names = list(_table()["StudyInstanceUID"])
    with pytest.raises(ValueError, match="1 of 8 table rows"):
        _resolve(monkeypatch, tmp_path, {"oof_fold0": names[:-1]})
