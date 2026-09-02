"""The two shapes MRNet arrives in, asked the same two questions."""

import io
import zipfile
from pathlib import Path

import numpy as np

from core.data.corpus.mrnet_corpus import PLANES
from core.data.corpus.mrnet_source import ExtractedSource, MRNetSource, ZippedSource

DEPTH, MATRIX = 6, 32
ROOT = "MRNet-v1.0"
CASE = "0000"


def _volume() -> bytes:
    """One plane's stack in the corpus's own format — a plain `.npy`, never a pickle."""
    stack = np.arange(DEPTH * MATRIX * MATRIX, dtype=np.uint8).reshape(DEPTH, MATRIX, MATRIX)
    buffer = io.BytesIO()
    np.save(buffer, stack)
    return buffer.getvalue()


def _zipped(tmp_path: Path) -> Path:
    """The Stanford download's shape: every volume under a single root directory inside one zip."""
    path = tmp_path / "MRNet-v1.0.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for plane in PLANES:
            archive.writestr(f"{ROOT}/train/{plane}/{CASE}.npy", _volume())
    return path


def _extracted(tmp_path: Path) -> Path:
    """The shape that is actually on this machine: unpacked once, 7.3 GB of loose arrays."""
    root = tmp_path / "mrnet"
    for plane in PLANES:
        volume = root / "train" / plane / f"{CASE}.npy"
        volume.parent.mkdir(parents=True, exist_ok=True)
        volume.write_bytes(_volume())
    return root


def test_at(tmp_path: Path):
    """The reader is chosen by what the path IS, not by what it is called.

    A directory handed to `zipfile.ZipFile` raises `PermissionError: [Errno 13]` — a message about the
    wrong thing entirely, and it cost a decode its launch.
    """
    assert isinstance(MRNetSource.at(_zipped(tmp_path)), ZippedSource)
    assert isinstance(MRNetSource.at(_extracted(tmp_path)), ExtractedSource)


def test_zipped_source_names(tmp_path: Path):
    """Every member of the archive, named the way `read` will take it back."""
    names = ZippedSource(_zipped(tmp_path)).names()

    assert set(names) == {f"{ROOT}/train/{plane}/{CASE}.npy" for plane in PLANES}


def test_zipped_source_read(tmp_path: Path):
    """A member's bytes come back whole, without extracting the archive."""
    assert ZippedSource(_zipped(tmp_path)).read(f"{ROOT}/train/{PLANES[0]}/{CASE}.npy") == _volume()


def test_extracted_source_names(tmp_path: Path):
    """Names are relative to the root and posix-joined, so `_members` parses either shape alike."""
    names = ExtractedSource(_extracted(tmp_path)).names()

    assert set(names) == {f"train/{plane}/{CASE}.npy" for plane in PLANES}


def test_extracted_source_read(tmp_path: Path):
    """A member's bytes come back whole, resolved under the root it was named against."""
    assert ExtractedSource(_extracted(tmp_path)).read(f"train/{PLANES[0]}/{CASE}.npy") == _volume()
