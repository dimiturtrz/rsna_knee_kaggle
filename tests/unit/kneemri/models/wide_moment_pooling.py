"""The parameterless wide pooling: the concatenation the frozen probe read, and its declared width."""

import torch

from kneemri.models.moment_pooling import MOMENTS as MOMENTS_PARTS
from kneemri.models.slice_pooling import MOMENTS, MOMENTS_WIDE, SLICE_POOLINGS
from kneemri.models.wide_moment_pooling import WideMomentPooling

DIM = 8
SLICES = 5


def _stack(slices: int = SLICES, dim: int = DIM) -> torch.Tensor:
    """A deterministic (b, s, g, d) block: unit tests here create every tensor they touch."""
    return torch.arange(2 * 3 * slices * dim, dtype=torch.float32).reshape(2, 3, slices, dim) / 100.0


def test_wide_moment_pooling_forward():
    """The concatenation the frozen probe read, at double width, with no mix and no parameters."""
    pooling = WideMomentPooling(dim=DIM)
    x = _stack()

    pooled = pooling.forward(x)

    assert pooled.shape == (2, 3, MOMENTS_PARTS * DIM), "the head is FED twice what flows in"
    assert torch.allclose(pooled[..., :DIM], x.mean(dim=2), atol=1e-4), "the first half IS the mean"
    assert torch.allclose(pooled[..., DIM:], x.std(dim=2, unbiased=False), atol=1e-4), (
        "the second half IS the spread, unprojected — moment_dense mixed it down and lost the reading"
    )
    assert not list(pooling.parameters()), (
        "no tensors, so the 12-epoch budget fits the head alone, and there is nothing to leave out of an "
        "optimiser group (knee-7gyw)"
    )

    single = _stack(slices=1)
    assert torch.isfinite(pooling.forward(single)).all(), "the biased estimator keeps one slice from nan"


def test_wide_moment_pooling_registered():
    """The widening is declared where the head is BUILT, since a width found in a tensor is found too late."""
    entry = SLICE_POOLINGS[MOMENTS_WIDE]
    assert entry.per_slice, "each slice is encoded alone, as it is for moments and max"
    assert entry.out_multiplier == MOMENTS_PARTS, "two moments, so the head is constructed at twice the width"
    assert entry.aggregator is not None, "a None aggregator means a bare amax, which is what this replaces"
    assert isinstance(entry.aggregator(DIM), WideMomentPooling)
    assert SLICE_POOLINGS[MOMENTS].out_multiplier == 1, "every pooling that preserves the width says so"
