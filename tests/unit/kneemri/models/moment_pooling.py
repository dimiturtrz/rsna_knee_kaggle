import torch

from kneemri.models.moment_pooling import MomentPooling
from kneemri.models.slice_pooling import MOMENTS, SLICE_POOLINGS

DIM = 8
SLICES = 5


def _stack(slices: int = SLICES, dim: int = DIM) -> torch.Tensor:
    """A batch that VARIES along the slice axis, so mean and spread are different readings of it."""
    return torch.arange(2 * 3 * slices * dim, dtype=torch.float32).reshape(2, 3, slices, dim) / 100.0


def test_moment_pooling_forward():
    """An untrained pooling IS the mean over slices, and the spread reaches the output once trained."""
    pooling = MomentPooling(dim=DIM)
    x = _stack()

    pooled = pooling.forward(x)

    assert pooled.shape == (2, 3, DIM), "the slice axis is reduced and the width the head reads survives"
    assert torch.allclose(pooled, x.mean(dim=2), atol=1e-4), (
        "zero-initialised on the spread half and the identity on the mean half, so step zero is the mean "
        "— an aggregator that starts at an opinion it has no evidence for is worse than the max it replaces"
    )

    with torch.no_grad():
        weight = pooling.mix.weight
        weight[:, DIM:] = 1.0
    spread_reaches_output = not torch.allclose(pooling.forward(x), x.mean(dim=2), atol=1e-4)
    assert spread_reaches_output, "with the spread half open the output must stop being the mean"

    single = _stack(slices=1)
    assert torch.isfinite(pooling.forward(single)).all(), (
        "one slice in a slot is a real shape (V-JEPA2's cache carries a slice axis of length one) and the "
        "biased estimator is what keeps it from being nan"
    )


def test_forward_cannot_see_slice_order():
    """Both moments are symmetric in the slice axis, so this pooling reads no ORDER along the stack."""
    pooling = MomentPooling(dim=DIM)
    with torch.no_grad():
        weight = pooling.mix.weight
        weight[:] = torch.linspace(-1.0, 1.0, DIM * 2 * DIM).reshape(DIM, 2 * DIM)
    x = _stack()

    shuffled = pooling.forward(x[:, :, [3, 1, 4, 0, 2]])

    assert torch.allclose(pooling.forward(x), shuffled, atol=1e-3), (
        "DepthAttention owns the order question; keeping them separate is what makes either readable"
    )


def test_registered_in_the_vocabulary():
    """The name reaches the model builder, because a pooling that only the parser knows is built as max."""
    assert SLICE_POOLINGS[MOMENTS].per_slice, "each slice is encoded alone, as it is for attend and max"
    aggregator = SLICE_POOLINGS[MOMENTS].aggregator
    assert aggregator is not None, "a None aggregator means a bare amax, which is the pooling this replaces"
    assert isinstance(aggregator(DIM), MomentPooling)
