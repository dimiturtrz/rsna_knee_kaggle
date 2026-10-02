from pathlib import Path

import numpy as np
import pydicom

from core.data.corpus.depth_match import DepthMatch, Plane
from tests.unit.conftest import write_series

SIZE = 32


def _one_byte_body(path: Path, pixels: np.ndarray) -> None:
    """The raw-corpus defect: a 16-bit header over exactly one byte per pixel."""
    data = pydicom.dcmread(str(path))
    data.PixelData = pixels.astype(np.uint8).tobytes()
    data.save_as(str(path))


def test_read(tmp_path: Path):
    """Declared pixels at their declared depth; a one-byte body under a 16-bit header as 8 bits; else None."""
    directory = write_series(tmp_path / "series", slices=3, size=SIZE)
    whole, defect, torn = sorted(directory.iterdir())
    bytes_ = np.random.default_rng(0).integers(0, 256, (SIZE, SIZE))
    _one_byte_body(defect, bytes_)
    data = pydicom.dcmread(str(torn))
    data.PixelData = data.PixelData[:100]
    data.save_as(str(torn))
    (tmp_path / "junk.dcm").write_bytes(b"not a dicom")

    plane = Plane.read(whole)
    assert plane is not None and plane.bits == 16
    assert np.array_equal(plane.pixels, pydicom.dcmread(str(whole)).pixel_array.astype(np.float32))
    repaired = Plane.read(defect)
    assert repaired is not None and repaired.bits == 8, "the body is read at the depth it was stored at"
    assert np.array_equal(repaired.pixels, bytes_.astype(np.float32))
    assert Plane.read(torn) is None, "a body that is neither depth is a failed decode, as before"
    assert Plane.read(tmp_path / "junk.dcm") is None


def test_matched():
    """Full-depth planes pass through untouched; a sub-depth one lands on its nearest full-depth scale."""
    generator = np.random.default_rng(0)
    anatomy = generator.normal(1000, 200, (SIZE, SIZE)).astype(np.float32)
    far = generator.normal(1000, 200, (SIZE, SIZE)).astype(np.float32)
    dark = Plane(anatomy / 16, 8)

    matched = DepthMatch.matched([Plane(far, 12), None, dark, Plane(anatomy, 12)])

    assert matched[0] is not None and np.array_equal(matched[0], far)
    assert matched[1] is None, "a failed decode stays failed for the neighbour fill"
    assert matched[2] is not None and np.allclose(matched[2], anatomy, atol=1e-2), "matched to index 3, not 0"
    assert DepthMatch.matched([]) == []


def test_rescaled():
    """Same anatomy on another scale is restored; unrelated pixels, a flat plane or a new shape are refused."""
    generator = np.random.default_rng(0)
    reference = generator.normal(1000, 200, (SIZE, SIZE)).astype(np.float32)

    restored = DepthMatch.rescaled(reference * 0.7 + 5, reference)

    assert restored is not None and np.allclose(restored, reference, atol=1e-2)
    assert DepthMatch.rescaled(generator.normal(0, 1, (SIZE, SIZE)).astype(np.float32), reference) is None
    assert DepthMatch.rescaled(np.zeros((SIZE, SIZE), np.float32), reference) is None
    assert DepthMatch.rescaled(reference[:16], reference) is None
