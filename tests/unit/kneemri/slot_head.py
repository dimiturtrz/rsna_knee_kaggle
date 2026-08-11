import torch

from kneemri.model import TARGETS
from kneemri.slot_head import REPORT_DIMS, SlotHead

SEED = 2026
DIM = 8


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
