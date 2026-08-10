from pathlib import Path

import torch

from kneemri.acquisition import Acquisition
from kneemri.slot_reader import SlotReader

from .conftest import write_series


def test_read(tmp_path: Path):
    """A series becomes a fixed-shape uint8 stack, physically cropped and percentile-normalised."""
    acquisition = Acquisition.probe("study", "series", write_series(tmp_path / "series", slices=10, size=128))
    stack = SlotReader(image_size=32).read(acquisition, slices=3)
    assert stack is not None
    assert stack.shape == (3, 32, 32)
    assert stack.dtype == torch.uint8
    assert int(stack.max()) > int(stack.min()), "a normalised stack must use its range"


def test_read_returns_none_when_the_series_is_empty(tmp_path: Path):
    """Nothing decodable means the slot is ABSENT, which the mask can say — not black, which it cannot."""
    empty = tmp_path / "empty"
    empty.mkdir()
    acquisition = Acquisition("study", "series", empty, (), "UNK", False, None, None, None)
    assert SlotReader().read(acquisition, slices=3) is None


def test_read_repeats_the_last_index_when_the_band_is_thin(tmp_path: Path):
    """Asking for more slices than the central band holds pads rather than failing."""
    acquisition = Acquisition.probe("study", "series", write_series(tmp_path / "thin", slices=2, size=32))
    stack = SlotReader(image_size=16).read(acquisition, slices=6)
    assert stack is not None
    assert stack.shape == (6, 16, 16)
