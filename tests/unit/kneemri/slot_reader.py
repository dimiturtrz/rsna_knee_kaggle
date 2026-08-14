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


def test_cropped(tmp_path: Path):
    """Everything expensive and nothing size-dependent — so ONE read serves every size, exactly.

    This is the whole reason the split exists: a submission carrying members that read 392, 336 and 224
    px currently decodes every DICOM three times, because `predict_groups` decodes once per pixel group.
    Reading once and resizing per target is only worth doing if it is EXACT, and byte-identity against a
    native read at each size is what makes it so — a downsample between sizes would instead move every
    member that already scores.
    """
    acquisition = Acquisition.probe("study", "series", write_series(tmp_path / "series", slices=10, size=128))
    reader = SlotReader(image_size=32)

    stack = reader.cropped(acquisition, 3)
    assert stack is not None

    for size in (32, 16):
        native = SlotReader(image_size=size).read(acquisition, 3)
        assert native is not None
        assert torch.equal(reader.sized(stack, size), native), f"one read must equal a native read at {size}"


def test_sized(tmp_path: Path):
    """The stack becomes uint8 on the requested grid, whatever the reader's own size is."""
    acquisition = Acquisition.probe("study", "series", write_series(tmp_path / "series", slices=10, size=128))
    reader = SlotReader(image_size=32)
    stack = reader.cropped(acquisition, 3)
    assert stack is not None

    small = reader.sized(stack, 16)

    assert small.shape == (3, 16, 16)
    assert small.dtype == torch.uint8
