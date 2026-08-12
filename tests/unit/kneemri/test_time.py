"""Deterministic test-time views: the identity must be exact, and a rotation must actually change."""

import torch

from kneemri.test_time import SpatialViews

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
