"""Deterministic test-time views: the identity must be exact, and a rotation must actually change."""

import pytest
import torch

from kneemri.inference import test_time
from kneemri.inference.test_time import SpatialViews

SEED = 2026


def test_apply():
    """The identity view is the input itself; a rotated view differs but keeps shape and dtype.

    Identity is returned UNTOUCHED rather than warped by zero degrees: a zero-degree grid_sample still
    resamples, so it would quietly blur every prediction we have ever made the moment views were
    enabled, and the single-view path must be a genuine no-op.
    """
    torch.manual_seed(SEED)
    images = torch.randint(0, 256, (2, 6, 4, 16, 16), dtype=torch.uint8)
    views = list(SpatialViews(degrees=(0.0, 6.0), shifts=((0.0, 0.0),)).apply(images))
    assert len(views) == 2
    assert torch.equal(views[0], images), "the identity view must be the input, not a resampling of it"
    assert views[1].shape == images.shape and views[1].dtype == torch.uint8
    assert not torch.equal(views[1], images), "a six-degree rotation has to change the pixels"


def test_len():
    """The view count is the multiplier on inference cost, so it is stated rather than discovered."""
    assert len(SpatialViews(degrees=(0.0, -6.0, 6.0), shifts=((0.0, 0.0),))) == 3
    assert len(SpatialViews(degrees=(0.0,), shifts=((0.0, 0.0), (0.02, 0.0)))) == 2


def test_warp_is_indifferent_to_the_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunking the warp is a memory fix and must not be an arithmetic one.

    The whole-batch warp allocated its float copy, its sampling grid and its result at the batch's own
    length — 6.2 GB for the 3,456 images a 48-study chunk holds, which is an out-of-memory the forward's
    row-halving retry never sees because it is raised before the forward (knee-c9q4). The fix is only
    legitimate if every view is bit-identical to the one the unchunked path produced, which is what this
    asserts: same pixels at a chunk that splits the batch as at one that cannot.
    """
    torch.manual_seed(SEED)
    images = torch.randint(0, 256, (3, 6, 4, 12, 12), dtype=torch.uint8)
    views = SpatialViews(degrees=(-6.0,), shifts=((0.0, 0.0),))

    monkeypatch.setattr(test_time, "WARP_CHUNK", 10_000)
    whole = next(iter(views.apply(images)))
    monkeypatch.setattr(test_time, "WARP_CHUNK", 7)
    pieces = next(iter(views.apply(images)))

    assert whole.shape == images.shape
    assert torch.equal(whole, pieces), "a chunk boundary changed a pixel, so the fix moved the answer"


def test_identity() -> None:
    """One unwarped view is the plain model; one ROTATED view is not, and length cannot tell them apart.

    Whoever asks whether an array computed elsewhere may stand in for this reading needs that
    distinction: `SpatialViews(degrees=(6.0,))` is also a single forward pass and a different function.
    """
    assert SpatialViews(degrees=(0.0,), shifts=((0.0, 0.0),)).identity
    assert not SpatialViews(degrees=(6.0,), shifts=((0.0, 0.0),)).identity
    assert not SpatialViews().identity
