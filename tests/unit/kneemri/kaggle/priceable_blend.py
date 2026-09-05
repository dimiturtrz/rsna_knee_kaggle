"""The blend is searched over members that held out a split and share one ruler."""

from pathlib import Path

import numpy as np

from kneemri.kaggle.priceable_blend import ANCHOR_BOARD, ANCHOR_OFFLINE, PriceableBlend
from kneemri.runs.holdout_record import HoldoutRecord

LABELS = 12
STUDIES = 40
WIDTH = 4
RNG = np.random.default_rng(2026)
STRONG = "strong_s2026"
"""The member whose predictions track the truth most closely, so forward selection takes it first."""
REMAPPED = "remapped_s2026"
"""Scored against a truth table of its own — the minority ruler the search has to drop."""


def _member(path: Path, truth: np.ndarray, noise: float) -> None:
    """One member's recorded predictions, and the checkpoint they were read from."""
    studies = [f"s{index}" for index in range(STUDIES)]
    predictions = truth * 2.0 + RNG.random((STUDIES, LABELS)) * noise
    HoldoutRecord(path).write(studies, predictions.astype(np.float64), truth.astype(np.float32), epoch=6)
    path.write_bytes(b"")


def _models(tmp_path: Path) -> Path:
    """Three members on the corpus ruler, one full-fit by name, one on a ruler of its own."""
    truth = (RNG.random((STUDIES, LABELS)) > 0.5).astype(np.float32)
    _member(tmp_path / f"{STRONG}.pt", truth, noise=0.5)
    _member(tmp_path / "middle_s2026.pt", truth, noise=1.5)
    _member(tmp_path / "weak_s2026.pt", truth, noise=3.0)
    _member(tmp_path / "shipped_ff.pt", truth, noise=0.5)
    _member(tmp_path / f"{REMAPPED}.pt", 1.0 - truth, noise=0.5)
    return tmp_path


def _blend(tmp_path: Path) -> PriceableBlend:
    """The search over that fixture, under the rule the kernels serve."""
    return PriceableBlend(_models(tmp_path), vote="rank", width=WIDTH)


def test_candidates(tmp_path: Path):
    """A member whose name marks it full-fit never enters the pool, however good its record."""
    names = sorted(path.stem for path in _blend(tmp_path).candidates())

    assert names == ["middle_s2026", REMAPPED, STRONG, "weak_s2026"], "shipped_ff is out on its name alone"


def test_on_the_corpus_ruler(tmp_path: Path):
    """The majority truth table is the ruler, so a re-mapped member is dropped without being named."""
    blend = _blend(tmp_path)

    kept = sorted(path.stem for path in blend.on_the_corpus_ruler(blend.candidates()))

    assert kept == ["middle_s2026", STRONG, "weak_s2026"], "the minority ruler goes, chosen by count not by name"


def test_scored(tmp_path: Path):
    """A member that tracks the truth scores above chance on the studies it held out."""
    blend = _blend(tmp_path)

    assert 0.5 < blend.scored([tmp_path / f"{STRONG}.pt"]) <= 1.0, "predictions built from the truth beat chance"


def test_chosen(tmp_path: Path):
    """Forward selection opens on the strongest member and never exceeds its width."""
    history = _blend(tmp_path).chosen()

    assert history, "a pool of three comparable members produces at least one seat"
    assert history[0][0] == STRONG, "the first seat is the best single member"
    assert len(history) <= WIDTH, "the width is a cap, not a target"


def test_fixed(tmp_path: Path):
    """A named set is scored as one blend, which is how a kernel is priced under the rule it serves."""
    priced = _blend(tmp_path).fixed([STRONG, "middle_s2026"])

    assert len(priced) == 1, "a fixed set is one row, not one row per member"
    assert 0.5 < priced[0][1] <= 1.0, "the pair scores above chance"


def test_predicted(tmp_path: Path):
    """The band brackets the anchor and widens on either side of it."""
    low, high = _blend(tmp_path).predicted(ANCHOR_OFFLINE)

    assert low == high == ANCHOR_BOARD, "an offline equal to the anchor predicts the anchor board exactly"


def test_report(tmp_path: Path):
    """The report closes on a prediction, and says out loud that it is direction only."""
    lines = _blend(tmp_path).report()

    assert lines[-1].startswith("predicted board "), "the last line is the number the search exists to produce"
    assert "DIRECTION ONLY" in lines[-1], "the slope's own verdict travels with every prediction it makes"
