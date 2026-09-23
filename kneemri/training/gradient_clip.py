"""A ceiling on the size of one optimiser step's gradient, so a late outlier batch cannot move far.

It lives beside `Trainer` rather than inside it for the reason `accumulation.py` does: `trainer.py` sits
against the god-file ceiling, and this answers on its own — one float, one call, no corpus and no card.

WHY THE PROJECT HAS NEVER HAD ONE. `clip_grad` appears nowhere in this tree; every number in the ledger
was measured with an unbounded step. That is not a decision anybody recorded, it is the reference recipe
this project started from, and the recipe has since moved: the backbone rate went 8e-6 -> 4e-5 and the
run that exposed the gap, `native224_long`, slid −0.0165 over twelve extra epochs while its training loss
kept falling. A step whose norm is set by whichever batch happened to be drawn is the standard mechanism
behind exactly that shape, and it is the standard answer to it.

WHAT IT IS NOT. It is not a regulariser on the WEIGHTS — `--weight-decay` is that, and the two are
independent axes. This bounds the DISTANCE one update may travel; decay bounds where the walk may end up.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

UNCLIPPED = 0.0
"""No ceiling at all — the unbounded step every number in the ledger was measured under, and therefore
the default. A norm of zero is meaningless as a ceiling (it would zero every gradient), which is what
makes it a safe spelling for "off"."""


@dataclass(frozen=True)
class GradientClip:
    """The maximum global gradient norm one optimiser step is allowed to carry.

    A NULL POLICY OBJECT rather than a branch in `fit_epoch`, like `Mixup` and `Accumulation` beside it:
    the trainer holds one of these always and calls it always, so the hottest loop this project owns
    grows no `if` and there is no site at which a flag can be silently skipped. The branch that remains
    is INSIDE `applied`, where it costs one comparison and is the difference between doing nothing and
    walking every gradient in the model to compute a norm that is then multiplied by one.
    """

    max_norm: float = UNCLIPPED

    def applied(
        self, model: torch.nn.Module, optimiser: torch.optim.Optimizer, scaler: torch.amp.GradScaler
    ) -> Float[Tensor, ""]:
        """Clip this step's gradients in place, immediately before the optimiser steps, and return the
        norm they had BEFORE clipping.

        THE SCALER IS UNSCALED FIRST AND IT MUST BE. Under fp16 the loss is multiplied by a scale factor
        before the backward pass, so the gradients this sees are that many times too large and a norm
        computed on them is a fact about the scaler rather than about the model — the ceiling would then
        do nothing on a small scale and clip everything on a large one, drifting run to run as the scaler
        adapts. `GradScaler.unscale_` returns immediately when the scaler is disabled, which is the bf16
        default here, so the call is correct under both precisions and costs nothing under one of them.
        `scaler.step` skips its own unscaling once this has happened, so the step is unchanged.

        The norm is measured ALWAYS, clipped or not, because it is what the run record keeps per step: a
        run whose steps grew is otherwise indistinguishable from one whose data got harder. It stays on
        the device; the reader syncs once per epoch.
        """
        scaler.unscale_(optimiser)
        parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
        norm = torch.nn.utils.get_total_norm([parameter.grad for parameter in parameters if parameter.grad is not None])
        if self.max_norm:
            torch.nn.utils.clip_grads_with_norm_(parameters, self.max_norm, norm)
        return norm
