from pathlib import Path

import numpy as np
import torch

from kneemri.acquisition import Acquisition
from kneemri.crop_centre import ImageCentre, TissueCentre
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


def _offset_stack() -> np.ndarray:
    """A stack whose only signal is a block well away from the image centre."""
    stack = np.zeros((4, 200, 200), np.float32)
    stack[:, 120:160, 130:170] = 1.0
    return stack


def test_crop_centre_default_is_unchanged():
    """The default reader crops EXACTLY where it always did — 99 GB of caches depend on it.

    Changing where the crop is taken changes the pixels, so a silent switch would invalidate every
    cached array and every member fitted on one, and no fingerprint would catch it: each is checked
    against its own reading.
    """
    stack = _offset_stack()
    reader = SlotReader(crop_mm=100.0, image_size=64)

    default = reader._crop(stack, spacing=1.0)
    centred = SlotReader(crop_mm=100.0, image_size=64, centre=ImageCentre())._crop(stack, spacing=1.0)

    assert np.array_equal(default, centred), "the default IS the image centre"
    tissue = SlotReader(crop_mm=100.0, image_size=64, centre=TissueCentre())._crop(stack, spacing=1.0)
    assert not np.array_equal(default, tissue), "and a tissue centre really does read different pixels"
    assert tissue.shape == default.shape, "the physical extent is the same either way"


def test_narrowed():
    """A tight extent is taken from the stack the wide read already produced, not decoded again.

    That is what lets SCALE be a slot axis rather than a second decode: 80mm is a subset of 130mm, so
    the multi-scale scheme costs encoder passes and no extra I/O.
    """
    stack = np.zeros((2, 200, 200), np.float32)
    reader = SlotReader(crop_mm=130.0, image_size=64)

    assert reader.narrowed(stack, 1.0, None).shape == stack.shape, "no extent named means untouched"
    assert reader.narrowed(stack, 1.0, 200.0).shape == stack.shape, "a WIDER extent cannot widen a crop"

    tighter = reader.narrowed(stack, 1.0, 80.0)
    assert tighter.shape[1] == 80, "at 1 mm per pixel an 80 mm extent is 80 pixels"
    assert tighter.shape[1] < stack.shape[1]
