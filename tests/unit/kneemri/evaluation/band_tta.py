"""Averaging one foreign family over its band: what may be averaged, and what the average is on.

The mirror of `kneemri/evaluation/band_tta.py`. Every fixture here is written into `tmp_path` by the
test itself, because the subject is the ALIGNMENT contract and a real pair of scans would prove only
that two files on disk happen to agree.
"""

from pathlib import Path

import numpy as np
import pytest

from kneemri.evaluation.band_tta import BandTTA

STUDIES = ("s0", "s1", "s2", "s3")


def _record(path: Path, predictions: np.ndarray, *, studies: tuple[str, ...] = STUDIES) -> Path:
    truth = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
    np.savez(path, studies=np.array(studies), predictions=predictions.astype(np.float32), truth=truth)
    return path


def test_aligned(tmp_path: Path):
    """Two servings of the same studies load as one subject; a different study list is refused, not mixed."""
    first = _record(tmp_path / "a.npz", np.arange(8, dtype=np.float32).reshape(4, 2))
    second = _record(tmp_path / "b.npz", np.arange(8, 0, -1, dtype=np.float32).reshape(4, 2))
    other = _record(tmp_path / "c.npz", np.arange(8, dtype=np.float32).reshape(4, 2), studies=("s0", "s1", "s2", "x"))

    tta = BandTTA.aligned([first, second])

    assert len(tta.bands) == 2, "both servings are held, as bands of one subject"
    with pytest.raises(ValueError, match="same studies"):
        BandTTA.aligned([first, other])


def test_combined(tmp_path: Path):
    """The average is in RANK space, so a band's score SCALE cannot outvote a band that disagrees with it."""
    ordering = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]], np.float32)
    same = _record(tmp_path / "a.npz", ordering)
    scaled = _record(tmp_path / "b.npz", ordering * 1000.0)

    averaged = BandTTA.aligned([same, scaled]).combined()

    assert averaged.tolist() == BandTTA.aligned([same]).combined().tolist(), "same ordering, thousandfold scale, one"
    assert averaged.max() == pytest.approx(1.0), "and the output is a rank, not a score"


def test_lines(tmp_path: Path):
    """The reading names every band it averaged AND the average, because one alone is not comparable."""
    first = _record(tmp_path / "a.npz", np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32))
    second = _record(tmp_path / "b.npz", np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], np.float32))

    lines = BandTTA.aligned([first, second]).lines()

    assert [line for line in lines if line.startswith("band ")] == ["band 0: 1.0000", "band 1: 0.0000"]
    assert "rank-mean over 2 bands" in lines[2], "and the average is reported beside them, never instead"
