"""Neighbouring slices mixed INSIDE the encoder's last blocks, through gates that start at zero.

Every block reads one slice alone; a slot's slices first meet after the encoder, in a pooling that is
order-blind by construction. So a tear crossing two touching slices is seen twice as two partial
structures and never once as one. Before each chosen block this adds, per channel, the previous and the
next slice's tokens to each slice's own. The gates start at zero, so a run starts exactly at the
control's function, and nothing costs a second forward pass.

Channel stacking (`adjacent224`) asked a related question and paid for contiguity with a third of the
span; this keeps every slice its own input and spends nothing on the span.
"""

from functools import partial
from typing import cast

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional


class SliceShift(nn.Module):
    """Per-channel gated neighbour tokens, hooked in front of each of `blocks`."""

    def __init__(self, blocks: list[nn.Module], dim: int) -> None:
        super().__init__()
        self.gates = nn.Parameter(torch.zeros(len(blocks), 2, dim))
        self.group = 1
        # A BOUND METHOD, so a deepcopy of the owning model (EMA, SWA) re-points each hook at the copy's
        # gates instead of leaving it reading the original's.
        for index, block in enumerate(blocks):
            block.register_forward_pre_hook(partial(self._hooked, index), with_kwargs=True)

    def _hooked(
        self, index: int, _: nn.Module, args: tuple[Tensor, ...], kwargs: dict[str, object]
    ) -> tuple[tuple[Tensor, ...], dict[str, object]]:
        if args:
            return (self.mixed(args[0], self.gates[index]), *args[1:]), kwargs
        hidden = cast("Tensor", kwargs["hidden_states"])
        return args, {**kwargs, "hidden_states": self.mixed(hidden, self.gates[index])}

    def mixed(self, tokens: Float[Tensor, "n t d"], gate: Float[Tensor, "2 d"]) -> Float[Tensor, "n t d"]:
        """Each slice's tokens plus its gated neighbours'; the stack's ends see zeros past the edge."""
        if self.group == 1:
            return tokens
        stack = tokens.reshape(-1, self.group, *tokens.shape[1:])
        before = functional.pad(stack[:, :-1], (0, 0, 0, 0, 1, 0))
        after = functional.pad(stack[:, 1:], (0, 0, 0, 0, 0, 1))
        return (stack + gate[0] * before + gate[1] * after).reshape(tokens.shape).to(tokens.dtype)
