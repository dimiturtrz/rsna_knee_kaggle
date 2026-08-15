"""Where a physical crop is taken about, on stacks whose true centre is known by construction."""

import numpy as np

from kneemri.corpus.crop_centre import Crop, ImageCentre, TissueCentre


def _offset_stack() -> np.ndarray:
    """A stack whose only signal is a block well away from the image centre."""
    stack = np.zeros((4, 200, 200), np.float32)
    stack[:, 120:160, 130:170] = 1.0
    return stack


def test_image_centre_of():
    """The geometric middle, whatever the tissue is doing."""
    assert ImageCentre().of(_offset_stack()) == (100, 100)


def test_tissue_centre_of():
    """The median of the per-slice centroids, and a fallback when there is no tissue at all."""
    row, column = TissueCentre().of(_offset_stack())
    assert 130 < row < 150, "inside the block, not at the frame's middle"
    assert 140 < column < 160

    empty = np.zeros((2, 40, 40), np.float32)
    assert TissueCentre().of(empty) == ImageCentre().of(empty), "no tissue falls back rather than raising"


def test_tissue_centre_centroid():
    """One slice's centroid, and None when the slice holds too little signal to locate."""
    stack = _offset_stack()

    found = TissueCentre().centroid(stack[0])
    assert found is not None
    assert 130 < found[0] < 150

    assert TissueCentre().centroid(np.zeros((40, 40), np.float32)) is None, "an empty slice abstains"


def test_tissue_centre_threshold():
    """Otsu splits a bimodal slice between its two modes, with no threshold anyone chose."""
    plane = np.zeros((100, 100), np.float32)
    plane[:, 50:] = 10.0

    threshold = TissueCentre().threshold(plane)

    assert 0.0 < threshold < 10.0, "between the background and the tissue mode"
    assert np.isinf(TissueCentre().threshold(np.zeros((10, 10), np.float32))), "a flat slice has no split"


def test_suffix():
    """The historic crop adds NOTHING to a cache's name, and every other one adds itself.

    Empty by design: those arrays are memory-mapped by running trainers and every member on disk was
    fitted on them, so the default must keep resolving to the file it always did. The two axes compose
    rather than shadow, or an 80mm tissue-centred cache would collide with an 80mm image-centred one.
    """
    assert Crop().suffix() == "", "the default keeps the historic cache name"
    assert Crop(mm=80.0).suffix() == "_80mm"
    assert Crop(centre="tissue").suffix() == "_tissue"
    assert Crop(mm=80.0, centre="tissue").suffix() == "_80mm_tissue"


def test_located():
    """The crop names its centring strategy, and builds the one it names."""
    assert isinstance(Crop().located(), ImageCentre)
    assert isinstance(Crop(centre="tissue").located(), TissueCentre)
