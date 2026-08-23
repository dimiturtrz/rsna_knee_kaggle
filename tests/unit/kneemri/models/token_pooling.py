import pytest
import torch

from kneemri.models.token_pooling import (
    CLS_MEAN,
    CLS_MEAN_FOCAL,
    FOCAL_FRACTION,
    MEAN,
    POOL_PARTS,
    SIDED,
    TOKEN_POOLINGS,
    ClassAndPatchMean,
    ClassAndPatchMeanAndTail,
    ClassAndSidedMean,
    PatchMean,
    TokenPooling,
)

SEED = 2026
TOKENS = 21
"""Five prefix rows and a SQUARE sixteen-patch grid, because `SIDED` splits that grid on its
midline column and a non-square count is refused rather than reshaped."""
DIM = 8
PREFIX = 5
"""A DINOv3-shaped prefix — one class token and four registers — because a prefix of one cannot tell
`tokens[:, 1:]` apart from `tokens[:, prefix:]`, which is the bug the field exists for."""


def _tokens() -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randn(2, TOKENS, DIM)


def test_token_pooling_reduce():
    """The base names a pooling and reduces nothing, so it must REFUSE rather than pool by accident.

    A name added to the vocabulary without a reduction is this project's recurring plumbing failure: the
    flag lands in the recorded config, the head is built for the declared width, and the forward pass
    quietly returns something else.
    """
    with pytest.raises(NotImplementedError, match="no reduction"):
        TokenPooling(parts=1).reduce(_tokens(), PREFIX)


def test_patch_mean_reduce():
    """The prefix rows are ABSENT from the average — the whole reason the `prefix` argument exists."""
    tokens = _tokens()
    mean = PatchMean().reduce(tokens, PREFIX)
    assert torch.allclose(mean, tokens[:, PREFIX:].mean(1))
    assert not torch.allclose(mean, tokens.mean(1)), "the prefix must not be averaged in"


def test_class_and_patch_mean_reduce():
    """The class token comes FIRST, because every member's head weights are laid out in that order."""
    tokens = _tokens()
    pooled = ClassAndPatchMean().reduce(tokens, PREFIX)
    assert pooled.shape == (2, DIM * POOL_PARTS[CLS_MEAN])
    assert torch.allclose(pooled[:, :DIM], tokens[:, 0])
    assert torch.allclose(pooled[:, DIM:], tokens[:, PREFIX:].mean(1))


def test_class_and_patch_mean_and_tail_reduce():
    """The tail is taken PER CHANNEL over the patch grid, not by selecting whole patches."""
    tokens = _tokens()
    patch = tokens[:, PREFIX:]
    focal = ClassAndPatchMeanAndTail().reduce(tokens, PREFIX)
    k = max(1, patch.shape[1] // FOCAL_FRACTION)

    assert focal.shape == (2, DIM * POOL_PARTS[CLS_MEAN_FOCAL])
    assert torch.allclose(focal[:, 2 * DIM :], patch.topk(k, dim=1).values.mean(1))
    assert not torch.allclose(focal[:, 2 * DIM :], focal[:, DIM : 2 * DIM]), "a tail equal to the mean is no tail"


def test_every_pooling_emits_the_parts_it_declares():
    """`POOL_PARTS` fixes the head's INPUT width while `reduce` produces its actual one.

    Derived from the same table so they cannot drift, and asserted by iteration so a pooling added later
    is covered without an edit here — the failure otherwise is a shape error deep in the first batch,
    after the backbone is fetched and the cache is mapped.
    """
    tokens = _tokens()
    for name, pooling in TOKEN_POOLINGS.items():
        assert pooling.reduce(tokens, PREFIX).shape == (2, DIM * pooling.parts), name
        assert POOL_PARTS[name] == pooling.parts, name

    assert set(POOL_PARTS) == {CLS_MEAN, CLS_MEAN_FOCAL, MEAN, SIDED}, "the vocabulary four call sites speak"


def test_class_and_sided_mean_reduce():
    """The two halves must SEPARATE the sides, and together must still be the mean the default emits.

    Both halves matter. A pooling whose halves do not differ on a field that differs by column has not
    given the head a side at all; a pair of halves that does not average back to the patch mean has
    taken information away rather than adding the contrast to it.
    """
    tokens = _tokens()
    pooled = ClassAndSidedMean().reduce(tokens, PREFIX)
    patch = tokens[:, PREFIX:]

    assert pooled.shape == (2, DIM * POOL_PARTS[SIDED])
    assert torch.allclose(pooled[:, :DIM], tokens[:, 0]), "the class token comes first, as in every other pooling"
    left, right = pooled[:, DIM : 2 * DIM], pooled[:, 2 * DIM :]
    assert torch.allclose((left + right) / 2, patch.mean(1), atol=1e-6), "the halves partition the grid"
    assert not torch.allclose(left, right), "halves that agree carry no side"


def test_class_and_sided_mean_reduce_separates_the_columns():
    """A field that is +1 on the left columns and -1 on the right must read as exactly that.

    The property the whole pooling exists for, on an input where the answer is known by construction:
    under any collapsing pooling this field averages to zero and is invisible.
    """
    side = 4
    grid = torch.ones(1, side, side, DIM)
    grid[:, :, side // 2 :] = -1.0
    tokens = torch.cat([torch.zeros(1, PREFIX, DIM), grid.flatten(1, 2)], dim=1)

    pooled = ClassAndSidedMean().reduce(tokens, PREFIX)

    assert torch.allclose(pooled[:, DIM : 2 * DIM], torch.ones(1, DIM))
    assert torch.allclose(pooled[:, 2 * DIM :], -torch.ones(1, DIM))
    assert torch.allclose(tokens[:, PREFIX:].mean(1), torch.zeros(1, DIM)), "the mean this pooling replaces is blind"


def test_class_and_sided_mean_reduce_refuses_a_non_square_grid():
    """A token count with no square grid has no midline, so it must CRASH rather than reshape.

    The silent alternative is the failure this project keeps paying for: a flag lands in the recorded
    config, the head is built for the declared width, and the forward pass pools something else.
    """
    with pytest.raises(ValueError, match="not a square grid"):
        ClassAndSidedMean().reduce(torch.randn(2, PREFIX + 12, DIM), PREFIX)
