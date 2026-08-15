import torch

from kneemri.models.slice_attention import SliceAttention

DIM = 8
BATCH, SLOTS, GROUP = 2, 3, 4


def test_forward():
    """Pools the slice axis away, with weights that are a distribution over slices.

    The contract that matters is that it is a WEIGHTED MEAN: the output of a slot whose slices are all
    identical must be that same vector, whatever the weights happen to be. A max would also pass that,
    which is the point — this has to be a drop-in for the axis it replaces.
    """
    torch.manual_seed(2026)
    pool = SliceAttention(DIM).eval()
    x = torch.randn(BATCH, SLOTS, GROUP, DIM)

    out = pool.forward(x)

    assert out.shape == (BATCH, SLOTS, DIM), "the slice axis is pooled away"
    same = torch.randn(BATCH, SLOTS, 1, DIM).expand(-1, -1, GROUP, -1).contiguous()
    assert torch.allclose(pool.forward(same), same[:, :, 0], atol=1e-5), "identical slices must pool to themselves"


def test_forward_is_a_convex_combination_of_its_slices():
    """The output lies within the hull of the slices it pooled, and starts as their mean.

    This is the contract that separates an attention pooling from the alternatives: a SUM would leave
    the hull, and a max would sit on its boundary. Being inside it is what lets the module keep more
    than one slice, which is the entire reason for replacing the max.

    Selectivity is deliberately not asserted here. At initialisation the gate is near-uniform, so this
    IS approximately a mean — the right default for an untrained module, since an attention starting
    somewhere arbitrary would be worse than the max it replaces. Which slice it comes to prefer is a
    property of training, and faking it by hand-setting weights through a LayerNorm tests the fixture
    rather than the model.
    """
    torch.manual_seed(2026)
    pool = SliceAttention(DIM).eval()
    x = torch.randn(BATCH, SLOTS, GROUP, DIM)

    with torch.no_grad():
        out = pool.forward(x)

    assert (out <= x.amax(dim=2) + 1e-5).all(), "a weighted mean cannot exceed its largest slice"
    assert (out >= x.amin(dim=2) - 1e-5).all(), "nor fall below its smallest"
    assert torch.allclose(out, x.mean(dim=2), atol=1e-5), (
        "an untrained gate is EXACTLY the mean: the output layer is zero-initialised so the module starts "
        "with no opinion, rather than at an arbitrary one it has no evidence for"
    )
