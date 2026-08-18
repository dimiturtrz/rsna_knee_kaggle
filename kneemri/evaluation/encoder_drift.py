"""How far supervision actually moved the encoder away from the weights it was pretrained with.

The gate for knee-c5n, and it is a falsification rather than a warm-up. On-domain self-supervised
pretraining is worth building only if there is room for it to fill: if twelve epochs of report labels
already rewrote the representation, a differently-pretrained encoder would simply be overwritten by the
same fine-tune, and the whole axis is dead for the cost of an afternoon of CPU.

Both sides are already on disk. The member checkpoints hold the fine-tuned encoder under a `backbone.`
prefix, and the stock weights come from the same `AutoModel.from_pretrained` the trainer used. So this
is arithmetic over two state dicts: no GPU, no forward pass, no data.

The stock weights are passed IN rather than fetched here, for the reason `HoldoutPredictions.load`
records: a method that downloads 90 MB is a method that cannot be tested.
"""

import argparse
import logging
import re
from pathlib import Path

import torch
from torch import Tensor

from kneemri.models.backbone import ENCODERS, SMALL
from kneemri.models.model import COMPILED_PREFIX

logger = logging.getLogger(__name__)

BACKBONE_PREFIX = "backbone."
BLOCK_INDEX = re.compile(r"\.(\d+)\.")
STEM = -1
"""The group for every encoder tensor outside the repeated blocks — patch embedding, class token,
positional table, the final norm. Negative so it sorts before block 0 and can never collide with one."""

CONCENTRATION_BLOCKS = 4
"""How many of the deepest blocks the concentration ratio is read over.

Four because that is what `--unfreeze-depth` opens by default, so the ratio answers the question the
trainer actually poses: did the part we allowed to move carry the adaptation, or did it leak into
blocks that were meant to stay put."""


class EncoderDrift:
    """One member's encoder, measured against the pretrained weights it started from.

    Per block rather than per tensor, because the question is WHERE supervision spent its capacity, and
    a single scalar over the whole encoder cannot tell a re-weighted top from a rebuilt representation.
    """

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
        self.saved = torch.load(checkpoint, map_location="cpu", weights_only=True)

    def variant(self) -> str:
        """Which encoder size this checkpoint records, so the caller fetches the matching stock weights.

        Read back from the config for the same reason the design is: a member compared against the wrong
        backbone produces a drift profile that is entirely the difference between two architectures.
        """
        return str(self.saved.get("config", {}).get("variant", SMALL))

    def repository(self) -> str:
        """The Hugging Face id of the stock weights this member was fitted from."""
        return ENCODERS[self.variant()].repository

    def trained(self) -> dict[str, Tensor]:
        """The fine-tuned encoder tensors, keyed as the stock model keys them.

        The head is dropped rather than compared: it has no pretrained counterpart, was initialised
        randomly, and its movement is not evidence about the representation.
        """
        return {
            name.removeprefix(BACKBONE_PREFIX).removeprefix(COMPILED_PREFIX): value.float()
            for name, value in self.saved["model"].items()
            if name.startswith(BACKBONE_PREFIX) and value.is_floating_point()
        }

    def grouped(self, state: dict[str, Tensor]) -> dict[int, list[str]]:
        """Tensor names by block index, with everything outside the repeated blocks under `STEM`.

        The index is taken from the first integer path segment rather than from a naming convention,
        because DINOv2 keeps its blocks at `encoder.layer.N` and DINOv3 does not, and a rule that reads
        one of those two silently reports a single-block encoder for the other.
        """
        groups: dict[int, list[str]] = {}
        for name in state:
            found = BLOCK_INDEX.search(name)
            groups.setdefault(int(found.group(1)) if found else STEM, []).append(name)
        return groups

    def blocks(self, stock: dict[str, Tensor]) -> dict[int, float]:
        """Relative drift per block: the norm of the change over the norm of the pretrained weights.

        Relative, because absolute norms grow with width and depth and would rank a large encoder's
        blocks against a small one's by size alone. A tensor the stock model does not have is skipped
        rather than counted as infinite drift — that is a new parameter, not a moved one.
        """
        trained = self.trained()
        # A COMPARISON THAT MATCHES NOTHING IS A FAILURE, not a drift of zero. The first run of this
        # module reported `top-4 share of drift 0.000` for a compiled checkpoint whose every key carried
        # a prefix the stock model does not use — a number that reads as "supervision changed nothing".
        if not set(trained) & set(stock):
            message = (
                f"{self.checkpoint.name}: none of its {len(trained)} encoder tensors is named in the "
                f"{len(stock)} stock ones, so nothing was compared"
            )
            raise ValueError(message)
        drift: dict[int, float] = {}
        for block, names in self.grouped(trained).items():
            shared = [name for name in names if name in stock]
            if not shared:
                continue
            changed = sum(float((trained[name] - stock[name]).pow(2).sum()) for name in shared)
            original = sum(float(stock[name].pow(2).sum()) for name in shared)
            drift[block] = (changed**0.5) / (original**0.5) if original else 0.0
        return drift

    def concentration(self, drift: dict[int, float]) -> float:
        """The share of total drift sitting in the deepest `CONCENTRATION_BLOCKS` blocks.

        This is the number the gate turns on. Near 1.0 means supervision only re-weighted the top of a
        representation it could not afford to rebuild, and the room on-domain pretraining would fill is
        real. Spread evenly — near `CONCENTRATION_BLOCKS / n_blocks` — means the fine-tune rewrites the
        whole encoder anyway, and pretraining it differently first would be overwritten.
        """
        blocks = sorted(index for index in drift if index != STEM)
        if not blocks:
            return 0.0
        total = sum(drift[index] for index in blocks)
        deepest = sum(drift[index] for index in blocks[-CONCENTRATION_BLOCKS:])
        return deepest / total if total else 0.0

    def report(self, stock: dict[str, Tensor]) -> dict[int, float]:
        """Log the profile and hand it back, so a caller can compare two members without re-reading."""
        drift = self.blocks(stock)
        logger.info("%s (%s)", self.checkpoint.name, self.variant())
        for block in sorted(drift):
            label = "stem" if block == STEM else f"block {block:2d}"
            logger.info("  %-9s %.4f", label, drift[block])
        # THE MAGNITUDE IS REPORTED BEFORE THE SHARE, because it turned out to be the number that
        # decides this question: every member measured moves its most-adapted block by 0.2-2.1% of that
        # block's own norm. A share of a quantity that small says where a little went, not how much.
        logger.info("  largest block drift   %.4f", max(drift.values(), default=0.0))
        if not any(drift.values()):
            # LoRA is the honest zero: the base is frozen by construction and the adapters are separate
            # parameters, so there IS no drift to concentrate. Saying so beats printing a 0.000 share
            # that looks like a measurement of an adapted encoder.
            logger.info("  base weights untouched — adapters or a fully frozen encoder")
            return drift
        logger.info("  top-%d share of drift  %.3f", CONCENTRATION_BLOCKS, self.concentration(drift))
        return drift


def main() -> None:
    """Profile one or more member checkpoints against their own stock backbones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from transformers import AutoModel  # noqa: PLC0415 - a 90 MB import the unit gate must not pay for

    for path in arguments.checkpoints:
        drift = EncoderDrift(path)
        pretrained = AutoModel.from_pretrained(drift.repository()).state_dict()
        drift.report({name: value.float() for name, value in pretrained.items()})


if __name__ == "__main__":
    main()
