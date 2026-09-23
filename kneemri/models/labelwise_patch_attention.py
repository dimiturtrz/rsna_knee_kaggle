"""Per-diagnosis attention over every PATCH of a slot, added on top of the per-diagnosis slice pooling."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from core.findings import TARGETS
from kneemri.models.labelwise_attention import LabelwiseAttention
from kneemri.models.slice_attention import GATE


class LabelwisePatchAttention(nn.Module):
    """`LabelwiseAttention` plus a second, per-diagnosis softmax over all slice x patch tokens of a slot.

    WHAT NO OTHER POOLING CAN SELECT. Every other slice pooling receives each slice already collapsed to
    one vector by the token pooling (`cls_mean`), so the diagnosis-specific step only ever chooses among
    SLICES. `focal` and `sided` changed that collapse, but label-AGNOSTICALLY: one selection shared by all
    twelve findings, and both closed flat. Here each diagnosis runs a gated attention-MIL over the g*t
    patch tokens of the slot, so a meniscal tear and an effusion can read different regions of the same
    slice.

    IT STARTS AS `labelwise`. The patch reading enters through a projection whose weight and bias are
    zero, so at step zero the output is exactly the slice-pooled reading and any delta an arm measures is
    something training chose. The class-token summary is kept for the same reason: the patch term is
    ADDED to the existing path, never substituted for it.

    The score layer is zeroed as `LabelwiseAttention` zeroes its own, so the untrained patch softmax is
    the uniform mean over the slot's patches and no finding starts with an unearned preference.
    """

    def __init__(self, width: int, token_width: int, gate: int = GATE, n_out: int = len(TARGETS)) -> None:
        super().__init__()
        self.over_slices = LabelwiseAttention(width, gate, n_out)
        score = nn.Linear(gate, n_out)
        nn.init.zeros_(score.weight)
        nn.init.zeros_(score.bias)
        self.gate = nn.Sequential(nn.LayerNorm(token_width), nn.Linear(token_width, gate), nn.Tanh(), score)
        self.project = nn.Linear(token_width, width)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    @override
    def forward(self, x: Float[Tensor, "b s g w"], patches: Float[Tensor, "b s g t d"]) -> Float[Tensor, "b s o w"]:
        """One vector per diagnosis per slot: the slice-pooled reading plus that diagnosis's patch reading."""
        tokens = patches.flatten(2, 3)
        weights = self.gate(tokens).softmax(dim=2)
        attended = torch.einsum("bsno,bsnd->bsod", weights.to(tokens.dtype), tokens)
        return self.over_slices(x) + self.project(attended)
