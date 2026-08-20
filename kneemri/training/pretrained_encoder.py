"""Reading a domain-adapted checkpoint back into a stock backbone, loudly.

Its own module rather than a second class beside the objective because the reader outlives the writer:
the supervised recipe loads one of these long after the pretraining run is gone, and it is the only place
that can still catch the two ways of loading nothing.
"""

import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


class PretrainedEncoder:
    """The file `Pretrainer.save` wrote, read back into a stock backbone."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.saved = torch.load(path, map_location="cpu", weights_only=True)

    def variant(self) -> str:
        """Which encoder this file was adapted from, so a caller can refuse a mismatch."""
        return str(self.saved["variant"])

    def into(self, backbone: nn.Module, variant: str) -> nn.Module:
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
        weights = self.saved["encoder"]
        missing, unexpected = backbone.load_state_dict(weights, strict=False)
        # The mask token is the ONE key legitimately absent from a stock encoder's own reference: it is
        # created by the pretraining objective and unused at inference. Anything else missing means the
        # architectures disagree.
        surprises = [name for name in missing if "mask_token" not in name]
        if surprises or unexpected:
            message = (
                f"{self.path.name} does not fit this {variant}: {len(surprises)} weights unset, "
                f"{len(unexpected)} unrecognised (first: {(surprises + list(unexpected))[0]})"
            )
            raise ValueError(message)
        logger.info("encoder initialised from %s (%d tensors)", self.path.name, len(weights))
        return backbone
