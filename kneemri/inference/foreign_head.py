"""The pooling head a published foreign checkpoint carries, rebuilt from the shapes in its own state dict.

It is its own module because it is its own subject: the backbone is interchangeable across the six
archives (coatnet, convnextv2, effnetv2) and this head is the part they all share. It is also the part
that is genuinely unlike ours — a slot-max member commits to ONE window for all twelve findings, and this
gives every finding its own distribution over the windows.
"""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

ATTENTION_WIDTH = 256
"""Hidden units in the published head's attention MLP, fixed by the shapes in every checkpoint's own
state dict (`att.0.weight` is 256 x F). Named rather than inlined because it is THEIR constant, not ours."""


class ForeignHead(nn.Module):
    """The published per-finding attention pool: each finding attends over the windows on its own.

    Not a mean and not a max. Every finding gets its own distribution over the study's windows, which is
    the mechanism our members do not have.
    """

    def __init__(self, dim: int, findings: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Sequential(nn.Linear(dim, ATTENTION_WIDTH), nn.Tanh(), nn.Linear(ATTENTION_WIDTH, findings))
        self.clsW = nn.Parameter(torch.zeros(findings, dim))
        self.clsb = nn.Parameter(torch.zeros(findings))

    @override
    def forward(self, features: Float[Tensor, "b k f"]) -> Float[Tensor, "b n"]:
        """One logit per finding, from that finding's own weighted mean over the windows."""
        hidden = self.norm(features)
        weights = torch.softmax(self.att(hidden), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", weights, hidden)
        return (pooled * self.clsW).sum(-1) + self.clsb

    @staticmethod
    def weights(published: dict[str, Tensor]) -> dict[str, Tensor]:
        """The head's parameters under the names this class gives them.

        Their `att` is a four-entry Sequential (Linear, Tanh, Dropout, Linear) and ours is three, dropout
        being a training-time layer with no parameters. So `att.3.*` there is `att.2.*` here, and loading
        without the rename fails loudly rather than silently — which is the only reason this is a rename
        and not a reshaping of our own module to match a dropout we do not want at inference.
        """
        renamed = {"att.3.weight": "att.2.weight", "att.3.bias": "att.2.bias"}
        return {renamed.get(name, name): value for name, value in published.items() if not name.startswith("backbone.")}
