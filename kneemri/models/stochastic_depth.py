"""Put `DropPath` on a built trunk's residual blocks, at a rate that ramps linearly with depth.

WHY THIS IS A MODULE AND NOT A `from_pretrained` KEYWORD. `Dinov2Config` does carry `drop_path_rate`,
and `AutoModel.from_pretrained(where, drop_path_rate=r)` would reach it in one line. Two things are
wrong with that line. The first is that transformers builds `Dinov2DropPath(config.drop_path_rate)` at
EVERY layer with the SAME rate, while the recipe this mechanism has published effect sizes under ramps
the rate linearly from zero at the first block to the full rate at the last -- the early blocks carry
the features every later block reads, and dropping them at the full rate is why a flat schedule
underperforms its own ramped control. The second is that `Encoder` is a frozen module-level registry
shared by every consumer including the kernel, so a per-run number cannot live in it; threading one
through `load` would put a training concern in the seam that serving also crosses.

So the rate is applied to a trunk that is already built, by REPLACING the `nn.Identity` that
transformers leaves behind when the configured rate is zero. That keeps the change to training alone:
`DropPath` is a no-op in `eval()`, holds no parameters and registers no buffers, so a checkpoint trained
under it is byte-identical in shape and key set to one trained without it, and the member a rebuild
constructs is the same member. It is the same write-off `sharpness` carries.

THE WITNESS IS THE RETURN VALUE, and it is not a nicety. `applied` finds its targets by attribute name,
so a trunk family that spells the attribute differently silently receives nothing -- and an arm whose
rate reached zero blocks is an ordinary run reporting under the stochastic-depth arm's name, with an
identical checkpoint, an identical score and an identical ledger row. Nothing downstream can tell those
apart. The count is the only place the question is answerable, which is why a nonzero rate that lands on
no block raises rather than warns.
"""

import logging

from torch import nn

from kneemri.models.drop_path import NO_DROP, DropPath

logger = logging.getLogger(__name__)

BRANCH = "drop_path"
"""The attribute a residual block hangs its stochastic-depth policy on.

`transformers.models.dinov2.modeling_dinov2.Dinov2Layer` spells it this way, and so do timm's ViT
blocks, which is what makes the walk generic over the two trunk families this project builds.
"""

NOTHING_TO_DROP = (
    "stochastic depth found no block carrying a `drop_path` attribute, so the rate reached nothing -- "
    "this trunk family spells its residual policy under another name"
)


class StochasticDepth:
    """The rate, and the walk that puts it on a trunk's blocks with a linear ramp over depth."""

    def __init__(self, rate: float = NO_DROP) -> None:
        self.rate = rate

    def applied(self, trunk: nn.Module) -> int:
        """Replace each block's `drop_path` with a ramped `DropPath`; answer how many were reached."""
        if not self.rate:
            return 0
        blocks = [module for module in trunk.modules() if hasattr(module, BRANCH)]
        if not blocks:
            raise ValueError(NOTHING_TO_DROP)
        # The last block carries the full rate and the first carries none, which is the published
        # schedule. A single-block trunk is the degenerate case and takes the full rate rather than
        # dividing by zero.
        last = max(len(blocks) - 1, 1)
        for depth, block in enumerate(blocks):
            block.drop_path = DropPath(self.rate * depth / last)
        logger.info("stochastic depth ramped to %g across %d blocks", self.rate, len(blocks))
        return len(blocks)
