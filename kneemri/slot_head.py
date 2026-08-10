"""Per-diagnosis attention over the acquisition slots of one study."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

_PRESENT = 0.5
"""Midpoint of the {0, 1} presence mask — a slot below it is absent and leaves the softmax."""

_ABSENT_LOGIT = -1e4
"""What an absent slot's attention logit becomes: negligible after softmax, finite in fp16."""


class SlotHead(nn.Module):
    """Per-diagnosis attention over the slot embeddings of one study.

    Each finding is read on particular sequences — cruciates sagittally, collateral ligaments and the
    meniscal body coronally, patellar cartilage axially — so pooling the slots identically would dilute
    the one carrying the evidence with the five that do not. Every diagnosis therefore gets its own
    query, and absent slots are masked out of the softmax so it renormalises over what the study
    actually contains rather than being fed a zero vector.

    The aggregation is deliberately this small. The label is attached to the study, so nothing in the
    supervision says which part of a study carries the finding, and a richer aggregator would have no
    signal to learn that from. Where the supervision is coarse, so is the head.
    """

    def __init__(self, dim: int, n_slot: int, n_out: int, hidden: int = 256, p: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

    @override
    def forward(self, x: Float[Tensor, "b s d"], mask: Float[Tensor, "b s"]) -> Float[Tensor, "b o"]:
        """Attend each diagnosis's query over the present slots and read out one logit per diagnosis."""
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden**0.5
        att = att.masked_fill(mask.unsqueeze(1) < _PRESENT, _ABSENT_LOGIT).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        readout = self.out
        return (ctx * readout.weight.unsqueeze(0)).sum(-1) + readout.bias
