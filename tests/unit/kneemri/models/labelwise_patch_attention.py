import torch

from core.findings import TARGETS
from kneemri.models.labelwise_attention import LabelwiseAttention
from kneemri.models.labelwise_patch_attention import LabelwisePatchAttention

SEED = 2026
WIDTH, TOKEN = 8, 4
BATCH, SLOTS, GROUP, PATCHES = 2, 3, 5, 9
OUT = len(TARGETS)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(BATCH, SLOTS, GROUP, WIDTH), torch.randn(BATCH, SLOTS, GROUP, PATCHES, TOKEN)


def test_forward():
    """The same `b s o w` a `labelwise` pooling hands up, so `SlotHead` consumes it unchanged."""
    torch.manual_seed(SEED)
    x, patches = _inputs()
    with torch.no_grad():
        out = LabelwisePatchAttention(WIDTH, TOKEN).eval().forward(x, patches)
        labelwise = LabelwiseAttention(WIDTH).eval().forward(x)
    assert out.shape == labelwise.shape == (BATCH, SLOTS, OUT, WIDTH)


def test_forward_starts_as_labelwise():
    """THE EQUALITY ORACLE: the zeroed projection makes an untrained patch reading add exactly nothing.

    Asserted against `LabelwiseAttention` itself under the same seed, because the claim is that this
    pooling starts where the incumbent starts, patches or no patches.
    """
    x, patches = _inputs()
    torch.manual_seed(SEED)
    patch_pool = LabelwisePatchAttention(WIDTH, TOKEN).eval()
    torch.manual_seed(SEED)
    labelwise = LabelwiseAttention(WIDTH).eval()
    with torch.no_grad():
        assert torch.equal(patch_pool.forward(x, patches), labelwise.forward(x))
        assert torch.equal(patch_pool.forward(x, patches * 10), labelwise.forward(x))


def test_forward_selects_patches_per_diagnosis_once_trained():
    """The mechanism: with a trained score and projection, findings read DIFFERENT regions of one slot.

    Without this the oracle above is satisfiable by a module that can never leave `labelwise`.
    """
    torch.manual_seed(SEED)
    pool = LabelwisePatchAttention(WIDTH, TOKEN).eval()
    x, patches = _inputs()
    with torch.no_grad():
        torch.nn.init.normal_(pool.gate[3].weight, std=1.0)
        torch.nn.init.normal_(pool.project.weight, std=1.0)
        out = pool.forward(x, patches) - pool.over_slices.forward(x)
        moved = pool.forward(x, patches.flip(dims=(3,)).roll(1, dims=3) * 2)

    assert not torch.allclose(out[:, :, 0], out[:, :, 1], atol=1e-3)
    assert not torch.allclose(moved, out + pool.over_slices.forward(x), atol=1e-3)
