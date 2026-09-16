"""Erasing a patch drawn from a map whose only nonzero cell is known, so the blanked block is known too."""

from pathlib import Path

import numpy as np
import pytest
import torch

from kneemri.training.patch_erasing import PatchErasing

STUDIES = 3
GRID = 4
SIZE = 8
CELL = SIZE // GRID
ROW, COLUMN = 1, 2
ALWAYS = 1.0
LIT = 200


def _pointed() -> PatchErasing:
    """Every study's map worth something only at one cell, so the draw cannot land anywhere else."""
    importance = np.zeros((STUDIES, GRID, GRID), dtype=np.float64)
    importance[:, ROW, COLUMN] = 1.0
    return PatchErasing(importance, ALWAYS)


def _images() -> torch.Tensor:
    return torch.full((STUDIES, 2, 3, SIZE, SIZE), LIT, dtype=torch.uint8)


def _written(maps: np.ndarray, names: list[str], destination: Path) -> None:
    np.savez(destination, studies=np.asarray(names), maps=maps)


def test_erased():
    """The drawn cell is blanked across every slot and slice, and nothing outside it is touched."""
    erased = _pointed().erased(_images(), np.arange(STUDIES), torch.Generator())

    block = erased[:, :, :, ROW * CELL : (ROW + 1) * CELL, COLUMN * CELL : (COLUMN + 1) * CELL]
    assert int(block.sum()) == 0, "the cell the whole map's mass sits on has to be the one that goes"
    assert int(erased.sum()) == int(_images().sum()) - int(block.numel()) * LIT


def test_erased_is_the_identity_at_the_null_rate():
    """Rate zero is byte-identical, so every ledger number keeps its meaning under this file."""
    images = _images()
    importance = np.zeros((STUDIES, GRID, GRID), dtype=np.float64)

    assert torch.equal(PatchErasing(importance, 0.0).erased(images, np.arange(STUDIES), torch.Generator()), images)


def test_loaded(tmp_path: Path):
    """The maps are joined to the corpus BY IDENTIFIER, not by position, and averaged over labels."""
    maps = np.zeros((2, 2, GRID, GRID), dtype=np.float32)
    maps[0, :, 0, 0] = 1.0
    maps[1, :, 3, 3] = 1.0
    destination = tmp_path / "sweep.npz"
    _written(maps, ["b", "a"], destination)

    erasing = PatchErasing.loaded(destination, ["a", "b"], ALWAYS)

    assert erasing.importance[0][3, 3] == pytest.approx(1.0), "study 'a' is the file's SECOND row"
    assert erasing.importance[1][0, 0] == pytest.approx(1.0)


def test_loaded_gives_an_unswept_study_the_mean_map(tmp_path: Path):
    """A study the sweep never saw erases a typical cell rather than dropping out of the augmentation."""
    maps = np.zeros((2, 1, GRID, GRID), dtype=np.float32)
    maps[0, :, 0, 0] = 1.0
    maps[1, :, 3, 3] = 1.0
    destination = tmp_path / "sweep.npz"
    _written(maps, ["a", "b"], destination)

    erasing = PatchErasing.loaded(destination, ["a", "b", "unswept"], ALWAYS)

    assert erasing.importance[2][0, 0] == pytest.approx(0.5)
    assert erasing.importance[2][3, 3] == pytest.approx(0.5)
