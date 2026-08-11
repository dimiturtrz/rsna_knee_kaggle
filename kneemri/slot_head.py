"""Per-diagnosis attention over the acquisition slots of one study."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

_PRESENT = 0.5
"""Midpoint of the {0, 1} presence mask — a slot below it is absent and leaves the softmax."""

_ABSENT_LOGIT = -1e4
"""What an absent slot's attention logit becomes: negligible after softmax, finite in fp16."""

_HIDDEN = 256
"""Width the slot embeddings are projected to before the per-diagnosis attention."""

_DROPOUT = 0.2
"""Applied to the attended context, where 4,407 studies meet a twelve-way readout."""


class SlotHead(nn.Module):
    """Per-diagnosis attention over the slot embeddings of one study.

    Each finding is read on particular sequences — cruciates sagittally, collateral ligaments and the
    meniscal body coronally, patellar cartilage axially — so pooling the slots identically would dilute
    the one carrying the evidence with the five that do not. Every diagnosis therefore gets its own
    query, and absent slots are masked out of the softmax so it renormalises over what the study
    actually contains rather than being fed a zero vector.

    Which sequence shows which finding is supplied as a prior on the attention logits rather than
    learned from scratch — see `anatomy`. The supervision here is one label vector per study and never
    says where in the study the finding was seen, so learning the mapping costs signal that is better
    spent elsewhere, and the mapping itself is not in doubt.

    The aggregation is deliberately this small. The label is attached to the study, so nothing in the
    supervision says which part of a study carries the finding, and a richer aggregator would have no
    signal to learn that from. Where the supervision is coarse, so is the head.
    """

    def __init__(self, dim: int, n_slot: int, n_out: int, prior: Float[Tensor, "o s"] | None = None) -> None:
        super().__init__()
        hidden, p = _HIDDEN, _DROPOUT
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        # A parameter, not a buffer: the anatomy is where the query should START looking, and the data
        # is allowed to overrule it. Zeros when absent, so the prior costs nothing when it is not used.
        self.prior = nn.Parameter(torch.zeros(n_out, n_slot) if prior is None else prior.clone())

    @override
    def forward(self, x: Float[Tensor, "b s d"], mask: Float[Tensor, "b s"]) -> Float[Tensor, "b o"]:
        """Attend each diagnosis's query over the present slots and read out one logit per diagnosis."""
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden**0.5
        att = att + self.prior.unsqueeze(0)
        att = att.masked_fill(mask.unsqueeze(1) < _PRESENT, _ABSENT_LOGIT).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        readout = self.out
        return (ctx * readout.weight.unsqueeze(0)).sum(-1) + readout.bias
