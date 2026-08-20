"""Reading a domain-adapted checkpoint back into a stock backbone, loudly.

Its own module rather than a second class beside the objective because the reader outlives the writer:
the supervised recipe loads one of these long after the pretraining run is gone, and it is the only place
that can still catch the two ways of loading nothing.
"""

import logging
import re
from pathlib import Path

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

BLOCK_INDEX = re.compile(r"\.(\d+)\.")
STEM_PREFIX = "embeddings."
ALL_BLOCKS = 0
"""`--encoder-adapt-depth 0`: take the whole adaptation, which is what a full load means."""
PAST_EVERY_BLOCK = 1 << 16
"""Where a tensor outside the repeated blocks sorts — the final norm belongs to the last block, so a
partial load leaves it stock rather than importing it under every depth."""


class PretrainedEncoder:
    """The file `Pretrainer.save` wrote, read back into a stock backbone."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.saved = torch.load(path, map_location="cpu", weights_only=True)

    def variant(self) -> str:
        """Which encoder this file was adapted from, so a caller can refuse a mismatch."""
        return str(self.saved["variant"])

    def upto(self, depth: int) -> dict[str, Tensor]:
        """The patch embedding plus the first `depth` blocks, or everything when `depth` is ALL_BLOCKS.

        WHY A PARTIAL LOAD IS A FIRST-CLASS OPERATION and not a hack. Two adaptations measured on
        2026-08-20 — six epochs at 1e-4 and at 1e-5 — scored 0.6902 and 0.6875 against a 0.8578 control,
        two numbers close enough to say the encoder's QUALITY was not what either arm measured. What
        they share is a rewritten TOP: block 11 moved 55% and 21% of its own norm where supervision has
        never moved it more than 0.8%. The head reads that block, so importing it hands the classifier a
        representation shaped for reconstruction.

        The argument this axis was built on only ever concerned the BOTTOM — blocks `--unfreeze-depth`
        freezes, which supervision provably cannot write into (drift exactly 0.0000) and therefore
        cannot repair either. Taking those blocks and leaving the rest stock is that argument stated
        exactly, instead of stated and then over-applied.

        The final layernorm is left stock with the blocks it follows: it is calibrated to the activations
        of block 11, and an adapted norm over stock blocks is a mismatch neither side asked for.
        """
        weights = self.saved["encoder"]
        if depth == ALL_BLOCKS:
            return weights
        return {
            name: value for name, value in weights.items() if name.startswith(STEM_PREFIX) or self._block(name) < depth
        }

    @staticmethod
    def _block(name: str) -> int:
        """The block a tensor belongs to; anything outside the repeated blocks sorts past every depth."""
        found = BLOCK_INDEX.search(name)
        return int(found.group(1)) if found else PAST_EVERY_BLOCK

    def into(self, backbone: nn.Module, variant: str, depth: int = ALL_BLOCKS) -> nn.Module:
        """Overwrite a stock backbone with the adapted weights, or refuse and say why.

        REFUSING IS THE POINT. `encoder_drift` reported a drift of exactly 0.000 for a compiled
        checkpoint whose every key carried a prefix the stock model does not use — a number that reads
        as "supervision changed nothing" and actually meant "nothing was compared". The same silence is
        available here and is worse: a supervised run that quietly kept its stock weights would report a
        score for domain-adaptive pretraining that measures the control.
        """
        if variant != self.variant():
            message = f"{self.path.name} adapted {self.variant()}, and this run asked for {variant}"
            raise ValueError(message)
        weights = self.upto(depth)
        withheld = set(self.saved["encoder"]) - set(weights)
        missing, unexpected = backbone.load_state_dict(weights, strict=False)
        # The mask token is the ONE key legitimately absent from a stock encoder's own reference: it is
        # created by the pretraining objective and unused at inference. A withheld key is the other, and
        # it is absent BY REQUEST — the depth this caller asked for. Anything else missing means the
        # architectures disagree.
        surprises = [name for name in missing if "mask_token" not in name and name not in withheld]
        if surprises or unexpected:
            message = (
                f"{self.path.name} does not fit this {variant}: {len(surprises)} weights unset, "
                f"{len(unexpected)} unrecognised (first: {(surprises + list(unexpected))[0]})"
            )
            raise ValueError(message)
        logger.info(
            "encoder initialised from %s (%d tensors, %d left stock)", self.path.name, len(weights), len(withheld)
        )
        return backbone
