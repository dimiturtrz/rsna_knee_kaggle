"""The join of a blocked sweep, over blocks that fit together and blocks that overlap."""

from pathlib import Path

import numpy as np
import pytest

from kneemri.evaluation.patch_map_blocks import PatchMapBlocks

LABELS = 12
GRID = 4


def _block(destination: Path, names: list[str]) -> Path:
    """One block file in the shape `patch_maps` writes, with each study's map carrying its own value."""
    maps = np.stack([np.full((LABELS, GRID, GRID), float(index)) for index in range(len(names))])
    np.savez(
        destination,
        studies=np.array(names),
        maps=maps,
        truth=np.zeros((len(names), LABELS)),
        tissue=np.ones((GRID, GRID)),
    )
    return destination


def test_joined(tmp_path: Path):
    """Two blocks concatenate in block order, so a study's row still belongs to that study."""
    blocks = [_block(tmp_path / "a.npz", ["s1", "s2"]), _block(tmp_path / "b.npz", ["s3"])]

    joined = PatchMapBlocks(blocks).joined()

    assert list(joined["studies"]) == ["s1", "s2", "s3"]
    assert joined["maps"].shape == (3, LABELS, GRID, GRID)
    assert joined["maps"][1].mean() == pytest.approx(1.0), "the second study keeps the second block row"
    assert joined["maps"][2].mean() == pytest.approx(0.0), "the third is the first row of its own block"


def test_joined_refuses_overlapping_blocks(tmp_path: Path):
    """A study two blocks both claim is a sweep whose offsets overlapped, not a study with two maps."""
    blocks = [_block(tmp_path / "a.npz", ["s1", "s2"]), _block(tmp_path / "b.npz", ["s2"])]

    with pytest.raises(ValueError, match="more than one block"):
        PatchMapBlocks(blocks).joined()
