import torch

from kneemri.findings import TARGETS
from kneemri.models.slot_head import REPORT_DIMS, MeanSlotHead, SlotHead

SEED = 2026
DIM = 8
SLOTS = 4
OUT = len(TARGETS)


def test_forward():
    """The head maps slot embeddings to one logit per diagnosis, renormalising over present slots."""
    torch.manual_seed(SEED)
    head = SlotHead(dim=DIM, n_slot=4, n_out=len(TARGETS))
    head.eval()
    with torch.no_grad():
        out = head.forward(torch.randn(3, 4, DIM), torch.ones(3, 4))
    assert out.shape == (3, len(TARGETS))


def test_forward_ignores_an_absent_slot():
    """Masking a slot out must give the same answer as never having supplied its embedding."""
    torch.manual_seed(SEED)
    head = SlotHead(dim=DIM, n_slot=3, n_out=len(TARGETS))
    head.eval()
    x = torch.randn(1, 3, DIM)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    corrupted = x.clone()
    corrupted[:, 2] = 1e3
    with torch.no_grad():
        assert torch.allclose(head.forward(x, mask), head.forward(corrupted, mask), atol=1e-4)


def test_report_vector():
    """The report readout is a study-level vector, and absent slots do not contribute to it.

    A report describes the whole study, so its target is a property of the study rather than of any
    one finding's attention — and an unfilled slot must not pull the average toward whatever it holds.
    """
    torch.manual_seed(SEED)
    head = SlotHead(dim=DIM, n_slot=4, n_out=len(TARGETS)).eval()
    x = torch.randn(2, 4, DIM)

    everything = torch.ones(2, 4)
    assert head.report_vector(x, everything).shape == (2, REPORT_DIMS)

    partial = torch.ones(2, 4)
    partial[:, -1] = 0.0
    dropped = head.report_vector(x, partial)

    x_changed = x.clone()
    x_changed[:, -1] = 99.0
    assert torch.allclose(dropped, head.report_vector(x_changed, partial), atol=1e-5), (
        "an absent slot's contents must not reach the report vector"
    )


def test_slot_head_attention():
    """Per-diagnosis weights that sum to one over the PRESENT slots and are zero on the absent ones."""
    torch.manual_seed(SEED)
    head = SlotHead(DIM, SLOTS, OUT).eval()
    features = torch.randn(2, SLOTS, DIM)
    mask = torch.ones(2, SLOTS)
    mask[0, -1] = 0.0

    weights = head.attention(head.proj(features) + head.slot_emb, mask)

    assert weights.shape == (2, OUT, SLOTS)
    assert torch.allclose(weights.sum(-1), torch.ones(2, OUT), atol=1e-5)
    assert weights[0, :, -1].max() < 1e-3, "an absent slot must leave the softmax"


def test_mean_slot_head_is_the_attention_at_its_limit():
    """With constant attention logits the two heads are the SAME function — the ablation is a limit.

    This is the trap the ablation was written to avoid, borrowed from a campaign that lost a whole
    conclusion to it: a knob whose zero lands between two behaviours produces a number that reads as a
    finding and is an artefact. Zeroing the query and the prior makes every slot's logit equal, so the
    softmax is uniform over the present slots, which is exactly what mean pooling computes. If the two
    disagree here, `mean` is a third mode and any denominator measured with it is meaningless.
    """
    torch.manual_seed(SEED)
    attend = SlotHead(DIM, SLOTS, OUT).eval()
    uniform = MeanSlotHead(DIM, SLOTS, OUT).eval()
    uniform.load_state_dict(attend.state_dict())
    with torch.no_grad():
        attend.query.zero_()
        attend.prior.zero_()
        uniform.query.zero_()
        uniform.prior.zero_()

    features = torch.randn(3, SLOTS, DIM)
    mask = torch.ones(3, SLOTS)
    mask[1, 2:] = 0.0

    with torch.no_grad():
        assert torch.allclose(attend.forward(features, mask), uniform.forward(features, mask), atol=1e-5)


def test_mean_slot_head_attention():
    """Uniform over the present slots, and exactly zero on the absent ones."""
    torch.manual_seed(SEED)
    head = MeanSlotHead(DIM, SLOTS, OUT).eval()
    mask = torch.ones(2, SLOTS)
    mask[0, 2:] = 0.0

    weights = head.attention(torch.randn(2, SLOTS, 256), mask)

    assert torch.allclose(weights.sum(-1), torch.ones(2, OUT), atol=1e-5)
    assert torch.allclose(weights[0, 0, :2], torch.full((2,), 0.5)), "uniform over the two present slots"
    assert weights[0, :, 2:].abs().max() == 0.0, "an absent slot contributes nothing"


def test_mean_slot_head_ignores_the_query_that_attention_uses():
    """The ablation must not secretly depend on the mechanism it removes.

    Equality at the limit is only half the proof: a head that still read the query would be measuring
    attention while claiming to have none.
    """
    torch.manual_seed(SEED)
    head = MeanSlotHead(DIM, SLOTS, OUT).eval()
    features = torch.randn(2, SLOTS, DIM)
    mask = torch.ones(2, SLOTS)

    with torch.no_grad():
        before = head.forward(features, mask)
        head.query.add_(torch.randn_like(head.query) * 10.0)
        head.prior.add_(torch.randn_like(head.prior) * 10.0)
        after = head.forward(features, mask)

    assert torch.allclose(before, after, atol=1e-6), "uniform pooling must not read the query or prior"
