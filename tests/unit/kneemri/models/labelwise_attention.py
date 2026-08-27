import torch

from core.findings import TARGETS
from kneemri.models.labelwise_attention import LabelwiseAttention
from kneemri.models.slice_attention import SliceAttention

SEED = 2026
DIM = 8
BATCH, SLOTS, GROUP = 2, 3, 5
OUT = len(TARGETS)


def test_forward():
    """The slice axis is replaced by a diagnosis axis: every finding gets its own view of the slot."""
    torch.manual_seed(SEED)
    pool = LabelwiseAttention(DIM).eval()
    with torch.no_grad():
        out = pool.forward(torch.randn(BATCH, SLOTS, GROUP, DIM))
    assert out.shape == (BATCH, SLOTS, OUT, DIM)


def test_forward_starts_as_the_mean_for_every_diagnosis():
    """THE EQUALITY ORACLE, and the reason the score layer is zeroed.

    At step zero each of the twelve softmaxes is uniform, so every diagnosis receives the average of the
    slot's slices — which is exactly what an untrained `SliceAttention` returns. An arm's delta is then
    entirely something training chose, and no part of it is paid for by an opinion about which slices
    matter that the module held before it saw a single study. Asserted against `SliceAttention` itself
    rather than against a hand-written mean, because the claim is that this pooling starts where the
    incumbent starts.
    """
    torch.manual_seed(SEED)
    x = torch.randn(BATCH, SLOTS, GROUP, DIM)
    with torch.no_grad():
        labelwise = LabelwiseAttention(DIM).eval().forward(x)
        shared = SliceAttention(DIM).eval().forward(x)

    assert torch.allclose(labelwise, shared.unsqueeze(2).expand_as(labelwise), atol=1e-5)


def test_forward_cannot_see_slice_order():
    """Permutation-invariant along depth, so a positive reading is per-diagnosis pooling and not order.

    Each slice is scored from that slice alone, so reversing depth permutes all twelve weight vectors
    with the slices and leaves every weighted sum untouched. Contiguity is `DepthAttention`'s axis and
    is deliberately kept out of this one — mixing them would leave a positive arm unattributable.
    """
    torch.manual_seed(SEED)
    pool = LabelwiseAttention(DIM).eval()
    with torch.no_grad():
        torch.nn.init.normal_(pool.gate[3].weight, std=0.5)
        x = torch.randn(BATCH, SLOTS, GROUP, DIM)
        assert torch.allclose(pool.forward(x), pool.forward(x.flip(dims=(2,))), atol=1e-5)


def test_forward_separates_the_diagnoses_once_the_score_is_trained():
    """The mechanism itself: with a trained score the twelve readings DIFFER, which is the whole point.

    Without this the zero-initialisation test above is satisfiable by a module that can never do
    anything else — a pooling permanently stuck at the mean would pass every other assertion here.
    """
    torch.manual_seed(SEED)
    pool = LabelwiseAttention(DIM).eval()
    with torch.no_grad():
        torch.nn.init.normal_(pool.gate[3].weight, std=1.0)
        out = pool.forward(torch.randn(BATCH, SLOTS, GROUP, DIM))

    assert not torch.allclose(out[:, :, 0], out[:, :, 1], atol=1e-3)
