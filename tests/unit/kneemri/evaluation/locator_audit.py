"""Landmark distances on regions whose centroids are known, and the RSNA grid writer on a cache it writes."""

from pathlib import Path

import numpy as np
import torch

from core.data.corpus.slots import SLOTS
from kneemri.evaluation.locator_audit import (
    CACHE,
    MILLIMETRES,
    PER_GRID,
    STUDIES,
    TILE_SLICES,
    Landmarks,
    LocatorAudit,
)
from kneemri.models.locator import CLASSES, Locator, SliceCorpus

SIDE = 28
CPU = torch.device("cpu")
TOLERANCE = 1e-4


def _square(top: int, left: int, tissue: int = 1) -> torch.Tensor:
    labels = torch.zeros(1, SIDE, SIDE, dtype=torch.long)
    labels[0, top : top + 6, left : left + 6] = tissue
    return labels


def test_empty() -> None:
    found = Landmarks.empty()
    assert set(found.errors) == set(range(1, len(CLASSES)))


def test_centroids() -> None:
    assert Landmarks.centroids(_square(2, 10).float()).tolist() == [[4.5, 12.5]]


def test_add() -> None:
    found = Landmarks.empty()
    found.add(_square(0, 0), _square(3, 4))
    found.add(torch.zeros(1, SIDE, SIDE, dtype=torch.long), _square(0, 0, tissue=2))
    assert abs(found.errors[1][0] - 5.0 * MILLIMETRES) < TOLERANCE
    assert (found.missed[2], found.present[2], found.extra[1]) == (1, 1, 0)


def test_lines() -> None:
    found = Landmarks.empty()
    found.add(_square(0, 0), _square(0, 0))
    assert [line.split(":")[0] for line in found.lines()] == [CLASSES[1]]


def test_landmarks(slice_corpus: SliceCorpus, tiny_locator: Locator) -> None:
    found = LocatorAudit(tiny_locator, CPU).landmarks(slice_corpus)
    assert sum(found.present.values()) > 0


def test_overlay() -> None:
    slices = np.full((20, SIDE, SIDE), 100, dtype=np.uint8)
    drawn = np.zeros((20, SIDE, SIDE), dtype=np.int64)
    drawn[:, 0, 0] = 1
    picture = LocatorAudit.overlay(slices, drawn)
    assert picture.shape == (SIDE, SIDE * len(TILE_SLICES), 3)
    assert picture[0, 0].tolist() == [177, 50, 50]
    assert picture[1, 1].tolist() == [100, 100, 100]


def test_rsna_grids(tmp_path: Path, tiny_locator: Locator) -> None:
    np.save(tmp_path / f"images_{CACHE}.npy", np.zeros((STUDIES + 1, len(SLOTS), 20, SIDE, SIDE), dtype=np.uint8))
    np.save(tmp_path / f"mask_{CACHE}.npy", np.ones((STUDIES + 1, len(SLOTS)), dtype=bool))
    written = LocatorAudit(tiny_locator, CPU).rsna_grids(tmp_path, SLOTS[0].name, tmp_path / "grids")
    assert len(written) == STUDIES // PER_GRID
    assert all(path.exists() for path in written)
