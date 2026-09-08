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

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor

from core import cli
from kneemri.models.backbone import ENCODERS, SMALL
from kneemri.models.model import COMPILED_PREFIX

logger = logging.getLogger(__name__)

BACKBONE_PREFIX = "backbone."
MEMBER_KEY = "model"
ADAPTED_KEY = "encoder"
"""The two shapes an encoder reaches this module in.

A fitted MEMBER holds the whole model — head included — with the encoder under `backbone.`. A
self-supervised ADAPTATION (`kneemri.training.pretrain`) holds nothing but the encoder, keyed exactly as
the stock model keys it. Reading only the first is how the SSL question — did pretraining write into the
blocks the fine-tune cannot reach — stayed unmeasured while the arm that depended on it ran.
"""
BLOCK_INDEX = re.compile(r"\.(\d+)\.")
STEM = -1
"""The group for every encoder tensor outside the repeated blocks — patch embedding, class token,
positional table, the final norm. Negative so it sorts before block 0 and can never collide with one."""

UNFREEZE_KEY = "unfreeze_last"
OFFSET_KEY = "unfreeze_offset"
"""The two fields `MemberRecord` writes into a checkpoint's config to record the window supervision was
allowed to move (`member_record.py`, "the value this run USED, never the module constant").

THE CONSTANT THAT USED TO SIT HERE WAS WRONG, and it is worth saying how rather than deleting it
quietly. It read `CONCENTRATION_BLOCKS = 4`, justified as "what `--unfreeze-depth` opens by default" —
a flag the trainer does not define. The real boundary is `--unfreeze-last`, whose default is 6, so the
ratio was read over the top FOUR blocks while supervision opened the top SIX: two opened blocks were
being counted as leak. It is `knee-jkqrg`, and the fix is not 4 -> 6. A legislated number cannot be
right for a checkpoint at all, because 274 of 296 arms take the default and the rest name their own
window, and `kneemri.evaluation` may not import `kneemri.training` to ask what the default is
(import-linter, "evaluation measures predictions"). The window belongs to the MEMBER and is recorded on
it; a checkpoint that does not record one is refused rather than assumed."""

CONSTRUCTED_SHARE = 0.9999
"""Where an in-window share stops being a measurement. Not a tolerance on a noisy quantity: outside a
supervised window the drift is arithmetically 0.0 over the same tensors, so the true value is exactly
1.0 and this absorbs only the float division."""


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
        recorded = self.saved.get("variant") or self.saved.get("config", {}).get("variant", SMALL)
        return str(recorded)

    def source(self) -> tuple[dict[str, Tensor], str]:
        """The state dict holding the encoder, and the prefix its keys carry."""
        if MEMBER_KEY in self.saved:
            return self.saved[MEMBER_KEY], BACKBONE_PREFIX
        return self.saved[ADAPTED_KEY], ""

    def repository(self) -> str:
        """The Hugging Face id of the stock weights this member was fitted from."""
        return ENCODERS[self.variant()].repository

    def trained(self) -> dict[str, Tensor]:
        """The fine-tuned encoder tensors, keyed as the stock model keys them.

        The head is dropped rather than compared: it has no pretrained counterpart, was initialised
        randomly, and its movement is not evidence about the representation.
        """
        state, prefix = self.source()
        return {
            name.removeprefix(prefix).removeprefix(COMPILED_PREFIX): value.float()
            for name, value in state.items()
            if name.startswith(prefix) and value.is_floating_point()
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

    def pairs(self, stock: dict[str, Tensor]) -> Iterator[tuple[int, float, float]]:
        """Per block: the squared norm of the change, and of the pretrained weights it is measured on."""
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
        for block, names in self.grouped(trained).items():
            shared = [name for name in names if name in stock]
            if not shared:
                continue
            yield (
                block,
                sum(float((trained[name] - stock[name]).pow(2).sum()) for name in shared),
                sum(float(stock[name].pow(2).sum()) for name in shared),
            )

    def blocks(self, stock: dict[str, Tensor]) -> dict[int, float]:
        """Relative drift per block: the norm of the change over the norm of the pretrained weights.

        Relative, because absolute norms grow with width and depth and would rank a large encoder's
        blocks against a small one's by size alone. A tensor the stock model does not have is skipped
        rather than counted as infinite drift — that is a new parameter, not a moved one.

        AND FOR EXACTLY THAT REASON IT MUST NOT BE READ ACROSS BLOCKS OF ONE ENCODER, which is what
        `changes` is for and why that method exists. Measured 2026-09-08 on stock `dinov2-small`: the
        per-block weight norm is not flat and does not trend smoothly — 69.8 at block 4, 204.6 at block
        11, with a 167.5 spike at block 3. So a CONSTANT absolute change prints here as a falling curve.
        Reading `adapt_small_e8_unitfix` off this method alone gave a 4.7x monotone early-heavy slope
        (0.1518 at block 4 down to 0.0321 at block 11) that is 1.8x and non-monotone in absolute terms,
        with block 11 above blocks 9 and 10 and block 3's "dip" entirely its own large norm. The profile
        was most of the way to a mechanism for a live arm's failure before the denominator was checked.
        """
        measured = self.pairs(stock)
        return {block: (changed**0.5) / (original**0.5) if original else 0.0 for block, changed, original in measured}

    def changes(self, stock: dict[str, Tensor]) -> dict[int, float]:
        """Absolute drift per block: the norm of the change, with no denominator at all.

        The companion `blocks` needs to be read against, and the only one of the two that compares
        blocks WITHIN one encoder honestly. It is meaningless ACROSS encoders — that is the direction
        `blocks` owns — so neither number is the real one and a shape claim needs both.
        """
        return {block: changed**0.5 for block, changed, _ in self.pairs(stock)}

    def opened(self) -> tuple[int, int] | None:
        """The block window supervision could move, as this member itself recorded it — or None.

        None is a refusal and not a zero: a checkpoint written before the fields existed, or by another
        project's trainer, cannot say which of its blocks were allowed to move, and a share read over a
        guessed window measures the guess.
        """
        config = self.saved.get("config", {})
        if UNFREEZE_KEY not in config:
            return None
        return int(config[UNFREEZE_KEY]), int(config.get(OFFSET_KEY) or 0)

    def concentration(self, drift: dict[int, float], window: tuple[int, int]) -> float:
        """The share of total drift sitting INSIDE the window supervision was allowed to move.

        Near 1.0 means the movement stayed where the trainer put it. Spread out — near the window's own
        share of the block count — means the encoder moved in blocks the trainer never opened, which for
        a member fitted from STOCK weights is impossible (`encoder_surgery.py` calls
        `requires_grad_(False)` on the rest) and therefore says the member did not start from stock.
        """
        blocks = sorted(index for index in drift if index != STEM)
        opened, offset = window
        hi = max(len(blocks) - offset, 0)
        inside = blocks[max(hi - opened, 0) : hi]
        total = sum(drift[index] for index in blocks)
        return sum(drift[index] for index in inside) / total if total else 0.0

    def constructed(self, drift: dict[int, float], window: tuple[int, int]) -> bool:
        """Is the share 1.000 because nothing outside the window moved AT ALL — i.e. by construction?

        This is the reading the old constant hid, and it is the same defect as `Ledger75Row.recut_blind`
        one axis over: a number whose population cannot express the question. A member fine-tuned from
        stock has its frozen blocks bit-identical to stock, so their drift is exactly 0.0 and the share
        is exactly 1.0 — a restatement of `--unfreeze-last`, never a measurement of where supervision
        chose to spend. The share becomes a real reading only when the frozen prefix DIFFERS from stock,
        which happens exactly when the member was fitted from a self-supervised checkpoint: then the
        outside-window drift is the PRETRAIN, and the ratio prices SSL against supervision rather than
        detecting a leak.
        """
        return self.concentration(drift, window) >= CONSTRUCTED_SHARE

    def report(self, stock: dict[str, Tensor]) -> dict[int, float]:
        """Log the profile and hand it back, so a caller can compare two members without re-reading."""
        drift = self.blocks(stock)
        absolute = self.changes(stock)
        logger.info("%s (%s)", self.checkpoint.name, self.variant())
        # BOTH COLUMNS, ALWAYS, because the shape of the relative one is partly the shape of the stock
        # norms it divides by and a reader comparing blocks needs the undivided number in the same view.
        for block in sorted(drift):
            label = "stem" if block == STEM else f"block {block:2d}"
            logger.info("  %-9s %.4f  (abs %8.3f)", label, drift[block], absolute[block])
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
        window = self.opened()
        if window is None:
            logger.info("  no %s recorded — this member cannot name a window to read a share over", UNFREEZE_KEY)
            return drift
        opened, offset = window
        logger.info("  last-%d (offset %d) share  %.3f", opened, offset, self.concentration(drift, window))
        if self.constructed(drift, window):
            logger.info("  ...which is 1.000 BY CONSTRUCTION: fitted from stock, so the frozen blocks cannot move")
        return drift


def main() -> None:
    """Profile one or more member checkpoints against their own stock backbones."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("checkpoints", type=Path, nargs="+")
    arguments = command.parsed()
    from transformers import AutoModel  # noqa: PLC0415 - a 90 MB import the unit gate must not pay for

    for path in arguments.checkpoints:
        drift = EncoderDrift(path)
        pretrained = AutoModel.from_pretrained(drift.repository()).state_dict()
        drift.report({name: value.float() for name, value in pretrained.items()})


if __name__ == "__main__":
    main()
