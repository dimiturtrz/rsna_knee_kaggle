"""Finding a mounted file without paying for the competition's study directories."""

from pathlib import Path

import pytest

from kneemri.kaggle.mounts import Mounts


def _mount(root: Path, *parts: str, name: str = "config.json") -> Path:
    holding = root.joinpath(*parts)
    holding.mkdir(parents=True, exist_ok=True)
    (holding / name).write_text("{}", encoding="utf-8")
    return holding


def test_walk(tmp_path: Path):
    """Study directories are pruned rather than filtered, and the depth bound stops the rest.

    Pruned rather than filtered is the whole point: `rglob` yields paths that CAN be filtered, but the
    walk has already descended and statted every DICOM by then, which is where 248 seconds went.
    """
    wanted = _mount(tmp_path, "models", "metaresearch", "dinov2", "PyTorch", "small", "1")
    _mount(tmp_path, "rsna-knee", "test_series", "study", "series")
    _mount(tmp_path, "a", "b", "c", "d", "e", "f", "g", "h", "i")

    assert Mounts(root=tmp_path).walk("config.json") == [wanted], "a six-deep encoder mount is reachable"
    assert Mounts(root=tmp_path, depth=2).walk("config.json") == [], "and the bound really does stop the walk"


def test_every(tmp_path: Path):
    """The answer is remembered, because the caller that hurt most asked once per member."""
    first = _mount(tmp_path, "weights", name="manifest.json")
    mounts = Mounts(root=tmp_path)

    assert mounts.every("manifest.json") == [first]

    _mount(tmp_path, "second", name="manifest.json")
    assert mounts.every("manifest.json") == [first], "walked once; twenty-two members must not pay twice"


def test_one(tmp_path: Path):
    """Exactly one, or a refusal — picking between two mounts loads weights a member never saw."""
    only = _mount(tmp_path, "weights")
    assert Mounts(root=tmp_path).one("config.json", "encoder") == only

    with pytest.raises(FileNotFoundError, match="no encoder"):
        Mounts(root=tmp_path).one("absent.json", "encoder")

    _mount(tmp_path, "other")
    with pytest.raises(FileNotFoundError, match="matches 2 mounts"):
        Mounts(root=tmp_path).one("config.json", "encoder")
