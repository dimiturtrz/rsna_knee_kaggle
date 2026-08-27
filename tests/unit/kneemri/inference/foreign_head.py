"""The published head's two contracts: it attends over windows, and it renames their dropout away."""

import torch

from core.findings import TARGETS
from kneemri.inference.foreign_head import ForeignHead

DIM = 8
WINDOWS = 5


def test_forward():
    """One logit per finding, from a distribution over windows rather than a sum of them."""
    head = ForeignHead(DIM, len(TARGETS))
    features = torch.randn(2, WINDOWS, DIM, generator=torch.Generator().manual_seed(0))

    logits = head.forward(features)

    assert logits.shape == (2, len(TARGETS))
    doubled = head.forward(torch.cat([features, features], dim=1))
    assert torch.allclose(logits, doubled, atol=1e-5), "a softmax over windows is invariant to duplicating them"


def test_weights():
    """Their four-entry `att` loads into our three-entry one, and the backbone is left to the backbone."""
    published = {
        "backbone.conv.weight": torch.zeros(1),
        "norm.weight": torch.zeros(DIM),
        "att.0.weight": torch.zeros(2, DIM),
        "att.3.bias": torch.zeros(len(TARGETS)),
    }

    weights = ForeignHead.weights(published)

    assert "backbone.conv.weight" not in weights, "the backbone's parameters are not the head's"
    assert "att.3.bias" not in weights, "their dropout-shifted index must not survive the rename"
    assert weights["att.2.bias"].shape == (len(TARGETS),)
    assert weights["norm.weight"].shape == (DIM,)
