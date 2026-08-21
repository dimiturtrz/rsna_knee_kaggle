"""Learned pooling over a slot's slices, in place of an elementwise maximum."""

from typing import override

from jaxtyping import Float
from torch import Tensor, nn

GATE = 64
"""Width of the gating projection. Small on purpose — see the class docstring.

Public because `DepthAttention` wraps this pooling and must default to the same width: a second copy of
the number is a second thing to keep in step, and the two modules are compared to each other."""


class SliceAttention(nn.Module):
    """Pool a slot's per-slice features by attention rather than by taking their maximum.

    Whole-stack reading — showing a step every cached slice instead of one three-slice window — is the
    largest single change measured in this project (+0.0147). It works by giving the model more of the
    stack, and then aggregates that with an elementwise maximum, which keeps whichever slice shouted
    loudest in each feature dimension and discards both which slice it was and everything the others
    said. This keeps the axis instead.

    The operator is the standard attention-MIL pooling (Ilse et al.): score each instance, softmax over
    the instances, take the weighted sum. `SlotHead` already does the same thing one level up over
    acquisition slots, so this is a familiar mechanism applied to the axis below it, and the published
    knee-MRI work that post-dates MRNet replaces exactly this max with exactly this attention (Medical
    Slice Transformer, 2025, on the same DINOv2 backbone we use).

    DELIBERATELY TINY: one gate of width 64 and a single shared query, so it adds a few thousand
    parameters rather than a transformer. This model is supervision-limited — 58 of 4,407 studies carry
    an image-read label — and forcing better attention has already measured NEUTRAL once, when the
    anatomical prior supplied correct slot attention and bought nothing. A per-diagnosis query over
    slices is the obvious next step and is deliberately not taken until a shared one earns it.
    """

    def __init__(self, dim: int, gate: int = GATE) -> None:
        super().__init__()
        # Zeroed so every slice scores identically at step zero and this pooling IS the mean of the
        # slices, which it then learns to deviate from. Default init gives arbitrary scores, so the
        # module would start at an opinion it has no evidence for - a worse default than the max it
        # replaces. Found by a test asserting "an untrained gate starts as a mean", which failed.
        # Built before the Sequential rather than reached through it: indexing a Sequential types as
        # Module | Tensor, so the initialisation could not be written against it.
        score = nn.Linear(gate, 1)
        nn.init.zeros_(score.weight)
        nn.init.zeros_(score.bias)
        self.gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, gate), nn.Tanh(), score)

    @override
    def forward(self, x: Float[Tensor, "b s g d"]) -> Float[Tensor, "b s d"]:
        """Weighted sum over the slice axis, weights summing to one per slot.

        PERMUTATION-INVARIANT, and that is not an accident of this implementation: the gate scores each
        slice from that slice alone, so shuffling the depth axis permutes the weights with the slices
        and leaves the sum untouched. `test_forward_cannot_see_slice_order` pins it.
        """
        weights = self.gate(x).softmax(dim=2)
        return (x * weights).sum(dim=2)
