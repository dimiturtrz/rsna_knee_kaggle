"""Low-rank adaptation: the pretrained weights never move, and the model still adapts.

THAT IS THE POINT, and it is not a memory technique that happens to freeze things. Our recipe takes the
last six blocks of an encoder trained on 1.7B images and rewrites them with gradients from 3,498
studies labelled out of radiology reports. Whatever those blocks learn about knees, they also forget
some of what made the encoder worth starting from — and nothing in our ledger can see that happening,
because the holdout is scored by the same fine-tuned model.

LoRA removes the trade. The base weight is frozen and stays bit-identical; adaptation lives entirely in
an added rank-r term. The pretrained function is still in there, exactly, and the update is constrained
to a subspace narrow enough that a small, noisy dataset cannot overwrite it. Every block can move a
little instead of six blocks moving without limit.

MEMORY AND SPEED ARE SIDE EFFECTS, and small ones here. Measured on this card at batch 2, whole-stack
336 px with checkpointing: six blocks unfrozen costs 2.50 GB and all twelve costs 3.22 GB, so nothing
about depth was memory-bound to begin with. The frozen base also needs no weight-gradient, worth about
a third of backward on an adapted layer — real, and far behind compile's +48%. Neither is a reason to
run this.

THE UPDATE STARTS AT EXACTLY ZERO. `up` is zero-initialised, so a freshly adapted model is
bit-identical to the pretrained one: the first steps cannot be a random perturbation of features that
took 1.7B images to learn.
"""

import math
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """A frozen `nn.Linear` plus a trainable rank-r correction.

    Wraps rather than replaces the base layer so the pretrained weight is still the weight: a member
    can be merged, inspected, or loaded back into the unadapted architecture by dropping the adapter
    keys, and nothing about the base's numerics changes.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)  # noqa: FBT003 - torch takes this positionally
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        # Kaiming on the way down and ZERO on the way up: the product is zero, so the adapted model
        # starts as the pretrained one exactly. Initialising both randomly would inject noise into
        # features that took 1.7B images to learn, at the step where the head knows least.
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    @override
    def forward(self, x: Float[Tensor, "... in"]) -> Float[Tensor, "... out"]:
        """The base layer's answer, corrected along the learned rank-r subspace."""
        return self.base(x) + self.up(self.down(x)) * self.scale

    @property
    def rank(self) -> int:
        """The subspace's width, read off the adapter rather than carried alongside it."""
        return int(self.down.out_features)

    def merged(self) -> nn.Linear:
        """The equivalent plain `nn.Linear`, with the correction folded into the base weight.

        Inference does not need the adapter's second matmul, and a merged member loads into the
        unmodified architecture — so a LoRA experiment cannot strand a checkpoint in a shape only this
        module can read.
        """
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + (self.up.weight @ self.down.weight) * self.scale)
            if self.base.bias is not None and merged.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged
