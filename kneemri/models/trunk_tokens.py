"""A trunk that does not speak the vision transformer's output contract, wearing it.

The head consumes (n, 1 + patches, dim) and is structural on purpose: any module callable as
`backbone(pixel_values=)` substitutes as a constructor argument, with no family test anywhere in it. The
two trunks that arrive short of that contract are adapted HERE rather than there, which is why this is a
module of its own and not a corner of the encoder table — the table says WHICH weights, this says how a
non-ViT set of them is made to answer the same questions.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn


@dataclass(frozen=True)
class TokenConfig:
    """The two fields the head reads off a backbone's config, for a trunk whose config lacks them.

    `ConvNextV2Config` carries `hidden_sizes` — one width per stage — and no `hidden_size`, and it has
    no notion of register tokens at all. The head derives its feature width and its prefix count from
    whatever the loaded weights say, deliberately, because the weights are the only authority on what
    they emit. So a trunk that does not answer those questions is given something that does, rather
    than the head being taught a second way to ask.
    """

    hidden_size: int
    num_register_tokens: int = 0


@dataclass(frozen=True)
class TokenOutput:
    """What a transformers backbone returns, for a trunk that returns a feature map instead."""

    last_hidden_state: Float[Tensor, "n t d"]

    @classmethod
    def from_spatial(cls, spatial: Float[Tensor, "n t d"]) -> "TokenOutput":
        """Spatial tokens with their own mean prepended, which is the slot a class token occupies.

        Mean-over-space is what a hierarchical trunk pools before its linear layer, so the prefix is not
        a placeholder: it holds the quantity a class token holds. It lives on the OUTPUT type because
        every trunk short of the contract needs it whatever layout it started from, and this is the type
        they all end at. Keeping it here is what holds `prefix_tokens` at 1 for every family, so the head's
        `tokens[:, 1:]` still selects exactly the spatial grid.
        """
        return cls(torch.cat([spatial.mean(dim=1, keepdim=True), spatial], dim=1))


class TrunkAsTokens(nn.Module):
    """A trunk that does not already speak the vision transformer's output contract, wearing it.

    The head consumes (n, 1 + patches, dim) — a token SEQUENCE with a class token in front — and
    teaching it anything else would put a family test in the one place this project has kept free of
    them: the head is structural on purpose, so any module callable as `backbone(pixel_values=)` can be
    substituted as a constructor argument. Two families arrive short of that contract, and they are
    short of DIFFERENT halves of it:

    - a ConvNeXt emits a feature MAP, (n, dim, h, w), and no prefix token;
    - a Swin emits a token SEQUENCE already, (n, t, d), but still no prefix token — it pools its final
      grid inside its own classifier, so what it hands back is spatial tokens and nothing else.

    Handing either one straight to the head puts a real spatial token where `tokens[:, 1:]` expects the
    prefix to have been, losing the grid's first cell with no shape error to announce it. The missing
    prefix is therefore shared and lives in `TokenOutput.from_spatial`; the only thing that differs is
    how the spatial tokens are reached, which is `to_spatial` — a strategy rather than a rank test,
    because a rank test would silently pick an arm for a trunk nobody had classified.
    """

    config: TokenConfig
    """Annotated at class level because `nn.Module.__getattr__` types every attribute it serves as
    `Tensor | Module`, so a reader of `backbone.config.hidden_size` would otherwise be reading a
    Tensor as far as the type checker is concerned."""

    def __init__(self, trunk: nn.Module, dim: int, to_spatial: Callable[[Tensor], Float[Tensor, "n t d"]]) -> None:
        super().__init__()
        self.trunk = trunk
        self.to_spatial = to_spatial
        self.config = TokenConfig(dim)

    @classmethod
    def from_map(cls, trunk: nn.Module, dim: int) -> "TrunkAsTokens":
        """For a trunk emitting (n, dim, h, w): flatten row-major, which is the ordering ViT patch
        tokens already carry, so the head's positional reading of the grid is unchanged."""
        return cls(trunk, dim, lambda emitted: emitted.flatten(2).transpose(1, 2))

    @classmethod
    def from_sequence(cls, trunk: nn.Module, dim: int) -> "TrunkAsTokens":
        """For a trunk emitting (n, t, d) with no prefix: the layout is already right and only the
        missing slot has to be supplied."""
        return cls(trunk, dim, lambda emitted: emitted)

    @property
    def is_gradient_checkpointing(self) -> bool:
        """What the trunk says, because the trunk is what checkpoints.

        `KneeModel.applied` reads this off the backbone to prove `--recompute-activations` LANDED, and
        `switch_coverage` refuses a run whose switch it cannot see. With the hook forwarded but the
        readback not, the flag was applied to the trunk and reported False from the wrapper — the
        plumbing failure the assert exists for, arriving as `asked True, the model shows False`.
        """
        return bool(getattr(self.trunk, "is_gradient_checkpointing", False))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, object]) -> None:
        """Forward transformers' checkpointing hook to the trunk this wrapper is wearing the contract of.

        `EncoderSurgery.recompute_activations` calls this on whatever it holds as a backbone, and until
        now that was always an HF model directly. A wrapped trunk is an `nn.Module` whose `__getattr__`
        raises for anything not a submodule, so `--recompute-activations` died with
        `'TrunkAsTokens' object has no attribute 'gradient_checkpointing_enable'` — an AttributeError at
        model construction, before a single batch. It was latent for as long as the wrapper existed: the
        one ConvNeXt arm that ran predates $THROUGHPUT and so never asked for checkpointing.

        Delegation rather than a family test in the surgery: the whole point of this class is that
        anything callable as `backbone(pixel_values=)` substitutes, and a caller that has to know which
        of the two it holds would put back the dispatch the wrapper exists to remove.
        """
        self.trunk.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)  # pyrefly: ignore[not-callable]

    @override
    def forward(self, pixel_values: Float[Tensor, "n c h w"]) -> TokenOutput:
        return TokenOutput.from_spatial(self.to_spatial(self.trunk(pixel_values=pixel_values).last_hidden_state))
