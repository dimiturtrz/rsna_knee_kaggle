"""Attention pooling over a slot's slices with ONE query per diagnosis, not one shared by all twelve."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from core.findings import TARGETS
from kneemri.models.slice_attention import GATE


class LabelwiseAttention(nn.Module):
    """`SliceAttention` with a separate set of pooling weights for each finding.

    THE ONE THING A SHARED QUERY CANNOT REPRESENT. A cruciate tear is two or three slices of signal; an
    osteoarthritic change spreads across most of the stack. One shared query has to serve both, so it is
    pulled twelve ways at once and settles near the compromise that satisfies none of them — which is
    exactly what `attend` measured: +0.0002 over an elementwise maximum, the null a shared query
    predicts for itself. That null is not evidence against attention pooling over slices; it is evidence
    that the axis was pooled with one weight when the supervision has twelve.

    `SlotHead` already gives every diagnosis its own query over the acquisition SLOTS, for the same
    reason and with the same operator. This is that mechanism applied one axis down, to the slices
    inside a slot, and it is the successor `SliceAttention`'s own docstring named and deferred.

    IT STARTS AS THE MEAN, for every diagnosis at once. The scoring layer is zeroed, so at step zero
    each of the twelve softmaxes is uniform and each diagnosis receives the same vector — the average of
    the slot's slices, which is precisely what an untrained `SliceAttention` returns. The twelve
    readings can then only differ by something training chose, and no part of an arm's delta is paid for
    by an opinion the module held before it saw data. `tests/unit/kneemri/models/labelwise_attention.py`
    pins it.

    It costs `n_out - 1` extra columns in one 64-wide linear — under a thousand parameters — because the
    gate is shared and only the SCORE is per diagnosis. The alternative, a full query per finding over
    the feature width, is twelve times the parameters for a task where 58 of 4,407 studies carry an
    image-read label.
    """

    def __init__(self, dim: int, gate: int = GATE, n_out: int = len(TARGETS)) -> None:
        super().__init__()
        # Zeroed for the reason `SliceAttention` zeroes its own score, once per diagnosis: an arbitrary
        # initialisation would start each finding at a different unearned opinion about which slices
        # matter, and the twelve-way spread of those opinions would then be charged to this mechanism.
        score = nn.Linear(gate, n_out)
        nn.init.zeros_(score.weight)
        nn.init.zeros_(score.bias)
        self.gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, gate), nn.Tanh(), score)

    @override
    def forward(self, x: Float[Tensor, "b s g d"]) -> Float[Tensor, "b s o d"]:
        """One pooled vector per diagnosis per slot, each a weighted sum over that slot's slices.

        The slice axis is REPLACED by a diagnosis axis rather than simply removed, which is what makes
        this pooling different in kind from the other three: downstream, each finding reads its own
        view of the slot instead of all of them sharing one. `SlotHead` takes that axis directly.

        PERMUTATION-INVARIANT along depth, like `SliceAttention` and for the same reason — each slice is
        scored from that slice alone, so shuffling depth permutes the twelve weight vectors with it and
        leaves every sum untouched. Contiguity is `DepthAttention`'s axis and is deliberately not mixed
        into this one, so a positive reading here is attributable to per-diagnosis pooling alone.
        """
        weights = self.gate(x).softmax(dim=2)
        return torch.einsum("bsgo,bsgd->bsod", weights, x)
