"""The adapter: the base weight never moves, and the model still changes."""

import torch
from torch import nn

from kneemri.lora import LoRALinear

SEED = 2026
IN, OUT, RANK = 8, 6, 2


def _base() -> nn.Linear:
    torch.manual_seed(SEED)
    return nn.Linear(IN, OUT)


def test_forward():
    """A fresh adapter is the identity, and the base weight is frozen from the moment it is wrapped.

    Zero-initialised `up` is what makes the first steps safe: the alternative injects noise into
    features that took 1.7B images to learn, at the point where the head knows least and the gradients
    are largest.
    """
    base = _base()
    original = base.weight.detach().clone()
    adapted = LoRALinear(base, RANK)
    x = torch.randn(4, IN)

    assert torch.allclose(adapted.forward(x), base(x)), "a fresh adapter must be the pretrained function exactly"
    assert not any(parameter.requires_grad for parameter in adapted.base.parameters()), "the base is frozen"
    assert adapted.down.weight.requires_grad and adapted.up.weight.requires_grad

    # After a step the ANSWER moves and the BASE does not — which is the entire proposition.
    optimiser = torch.optim.SGD([p for p in adapted.parameters() if p.requires_grad], lr=0.5)
    adapted(x).pow(2).mean().backward()
    optimiser.step()
    assert torch.equal(adapted.base.weight, original), "the pretrained weight must never move"
    assert not torch.allclose(adapted(x), base(x)), "the adapter must be able to change the answer"


def test_rank():
    """The subspace width is read off the adapter, not carried alongside it."""
    assert LoRALinear(_base(), RANK).rank == RANK


def test_merged():
    """A merged adapter is a plain Linear that computes the same function.

    This is what keeps a LoRA experiment from stranding a checkpoint in a shape only this module can
    read: inference needs no second matmul, and the result loads into the unmodified architecture.
    """
    adapted = LoRALinear(_base(), RANK)
    with torch.no_grad():
        adapted.up.weight.normal_(std=0.1)
    x = torch.randn(4, IN)

    merged = adapted.merged()
    assert isinstance(merged, nn.Linear)
    assert torch.allclose(merged(x), adapted(x), atol=1e-6)
