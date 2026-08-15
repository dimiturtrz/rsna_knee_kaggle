"""The cache key's partitions, which are the whole correctness story of caching a member's reading."""

from pathlib import Path

import numpy as np
import torch

from kneemri.inference.members import Window
from kneemri.inference.prediction_cache import PredictionCache
from kneemri.inference.test_time import SpatialViews

GROUP = 3
SLICES = 6
SIZE = 14


IDENTITY = SpatialViews(degrees=(0.0,), shifts=((0.0, 0.0),))


def _cache(tmp_path: Path, studies: list[str] | None = None) -> PredictionCache:
    return PredictionCache(tmp_path / "cached", studies or ["a", "b"], IDENTITY)


def _checkpoint(tmp_path: Path, name: str = "member.pt", payload: bytes = b"weights") -> Path:
    written = tmp_path / name
    written.write_bytes(payload)
    return written


def test_key(tmp_path: Path):
    """One partition per thing that moves the probabilities — a key missing any of them is a wrong array.

    This is the whole correctness story of the cache. The project's recurring defect is a value that
    travels per-run while a consumer reads a constant; a key that omits the window, the view set, the
    precision, the studies or the checkpoint's own bytes reproduces exactly that, silently, in the tool
    that decides which members to submit. Cheaper to enumerate the partitions than to debug one hit.
    """
    checkpoint = _checkpoint(tmp_path)
    window = Window(starts=[0], width=GROUP, size=SIZE)
    base = _cache(tmp_path).key(checkpoint, window, torch.float16)

    assert base == _cache(tmp_path).key(checkpoint, window, torch.float16), "the same reading must hit"
    assert base != _cache(tmp_path).key(checkpoint, Window(starts=[0, 3], width=GROUP, size=SIZE), torch.float16)
    assert base != _cache(tmp_path).key(checkpoint, Window(starts=[0], width=SLICES, size=SIZE), torch.float16)
    assert base != _cache(tmp_path).key(checkpoint, Window(starts=[0], width=GROUP, size=SIZE * 2), torch.float16)
    assert base != _cache(tmp_path).key(checkpoint, window, torch.bfloat16), "a bf16 member read as fp16 differs"
    assert base != _cache(tmp_path, ["a", "c"]).key(checkpoint, window, torch.float16), "another split, other rows"
    assert base != _cache(tmp_path, ["b", "a"]).key(checkpoint, window, torch.float16), "order aligns with truth"
    assert base != PredictionCache(tmp_path, ["a", "b"], SpatialViews()).key(checkpoint, window, torch.float16)


def test_key_misses_when_the_checkpoint_is_retrained_under_the_same_name(tmp_path: Path):
    """Name alone is not identity: an arm re-run to the same file is a different member.

    Re-running an arm under its own name is normal here — `resolution.sh` reproduces the member on disk —
    so this is the collision that would actually happen, not a hypothetical one.
    """
    checkpoint = _checkpoint(tmp_path)
    window = Window(starts=[0], width=GROUP, size=SIZE)
    before = _cache(tmp_path).key(checkpoint, window, torch.float16)

    retrained = _checkpoint(tmp_path, payload=b"different weights entirely")

    assert _cache(tmp_path).key(retrained, window, torch.float16) != before


def test_write(tmp_path: Path):
    """Values round-trip exactly, and the parts that identify them are stored beside the array."""
    cache = _cache(tmp_path)
    key = cache.key(_checkpoint(tmp_path), Window(starts=[0], width=GROUP, size=SIZE), torch.float16)
    values = np.array([[0.1, 0.9], [0.5, 0.25]])

    cache.write(key, values)

    stored = cache.read(key)
    assert stored is not None, "what was just written must read back"
    assert (stored == values).all()
    assert key in {path.read_text(encoding="utf-8") for path in (tmp_path / "cached").glob("*.json")}


def test_read(tmp_path: Path):
    """A reading never computed answers None, and a stored one whose sidecar disagrees also answers None.

    None rather than an exception for the miss, because a miss is the normal first run. But a file whose
    sidecar does not match is NOT a miss — it is a digest collision or a file copied between machines,
    and handing back another member's probabilities would be a wrong number rather than a slow one.
    """
    cache = _cache(tmp_path)
    key = cache.key(_checkpoint(tmp_path), Window(starts=[0], width=GROUP, size=SIZE), torch.float16)

    assert cache.read(key) is None, "nothing was ever written"

    cache.write(key, np.array([[0.3, 0.7]]))
    for sidecar in (tmp_path / "cached").glob("*.json"):
        sidecar.write_text('{"checkpoint": "somebody else.pt"}', encoding="utf-8")

    assert cache.read(key) is None, "a sidecar that disagrees must not answer"


def test_read_treats_a_killed_write_as_a_miss(tmp_path: Path):
    """A walk gets killed mid-member, and the store has to survive it as a MISS rather than an error.

    The back-fill over 69 checkpoints is hours of CPU and has already been killed once to re-order it.
    An array left without its sidecar used to make every later read of that key die on the sidecar —
    an interruption turned into a permanent failure for the one member it hit.
    """
    cache = _cache(tmp_path)
    key = cache.key(_checkpoint(tmp_path), Window(starts=[0], width=GROUP, size=SIZE), torch.float16)
    cache.write(key, np.array([[0.3, 0.7]]))
    for sidecar in (tmp_path / "cached").glob("*.json"):
        sidecar.unlink()

    assert cache.read(key) is None, "a half-written entry recomputes; it does not raise"

    cache.write(key, np.array([[0.3, 0.7]]))
    assert cache.read(key) is not None, "and the recomputation lands"
    assert not list((tmp_path / "cached").glob("*.pending")), "nothing is left half-committed"
