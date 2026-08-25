"""A fitted member, read back as an encoder another run can start from.

WHY THIS EXISTS. `--encoder-weights` has always been able to initialise a backbone from a file, and the
only files it could read were written by the spectral-MAE pretrainer. Twenty-three fitted members sat on
disk that no arm could start from, because a member checkpoint is a different shape: the whole model
under `model`, its backbone behind a `backbone.` prefix, and its adaptation living in LoRA adapters
rather than in the weights themselves. So every arm repeating an architecture paid for the first eight
epochs again — the ones that only re-learn what a member on disk already knows.

Converting rather than teaching the reader a second format is deliberate. `PretrainedEncoder.into`
refuses on a variant mismatch and on a key mismatch, and those refusals are the reason a warm start
cannot silently become a control run. A second reader would need its own copy of both, and the copy
that drifts is the one that stops refusing.

THE MERGE IS WHAT MAKES IT LOADABLE. A LoRA member's backbone is stock weights plus a rank-r term held
beside them, so its state dict does not fit the stock architecture at all — `into` would refuse it, and
correctly. `LoRALinear.merged` already folds the correction back into the base weight; this does the
same arithmetic over the state dict, because rebuilding the model to merge it would need the whole serve
path (design, pooling, slots) to answer a question about weights alone.
"""

import logging
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from core import cli

logger = logging.getLogger(__name__)

BACKBONE = "backbone."
COMPILED = "_orig_mod."
"""What `torch.compile` inserts. A checkpoint carrying it once reported an encoder drift of exactly
0.000 — every key compared against a name the stock model does not use, so nothing was compared."""
BASE_WEIGHT = ".base.weight"
BASE_BIAS = ".base.bias"
DOWN = ".down.weight"
UP = ".up.weight"
ADAPTER_ALPHA = 1.0
"""LoRA's alpha. Not a choice being re-made here: `low_rank` is called from two places and neither
passes one, so every member on disk was fitted at this value and `scale` is `alpha / rank`."""


class FittedEncoder:
    """The backbone of a member checkpoint, keyed and merged the way a stock encoder is."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.saved = torch.load(path, map_location="cpu", weights_only=True)

    def variant(self) -> str:
        """Which encoder this member was fitted on, so the written file can refuse a mismatch later."""
        config: dict[str, object] = self.saved["config"]
        return str(config["variant"])

    def cache(self) -> str:
        """The slice population the member saw, carried across so the warm start can name its domain."""
        config: dict[str, object] = self.saved["config"]
        return str(config.get("cache_studies", ""))

    def holdout(self) -> float:
        """What the member scored — the number that says whether this is worth starting from."""
        return float(self.saved["holdout"])

    def backbone_state(self) -> dict[str, Tensor]:
        """The backbone alone, keyed as `AutoModel.from_pretrained` keys it and with LoRA folded in.

        Three transformations, in the order the prefixes were applied: a compiled run's `_orig_mod.`
        comes off first, then the model's own `backbone.`, and only then can adapter names be
        recognised — a `.base.weight` is only a LoRA base once the prefixes above it are gone.
        """
        state: dict[str, Tensor] = self.saved["model"]
        stripped = {name.replace(COMPILED, ""): value for name, value in state.items()}
        backbone = {name[len(BACKBONE) :]: value for name, value in stripped.items() if name.startswith(BACKBONE)}
        return self.merge_adapters(backbone)

    @staticmethod
    def merge_adapters(state: dict[str, Tensor]) -> dict[str, Tensor]:
        """`base + up @ down * alpha / rank` for every adapted layer, and the rest untouched.

        The rank is read off the adapter's own shape rather than from the recipe. A member fitted at a
        rank the config never recorded still merges correctly, and a config that disagreed with the
        weights could not quietly rescale them.
        """
        merged: dict[str, Tensor] = {}
        for name, value in state.items():
            if name.endswith((DOWN, UP)):
                continue
            if name.endswith(BASE_BIAS):
                merged[f"{name[: -len(BASE_BIAS)]}.bias"] = value
            elif name.endswith(BASE_WEIGHT):
                stem = name[: -len(BASE_WEIGHT)]
                merged[f"{stem}.weight"] = value + FittedEncoder.correction(state, stem, value.dtype)
            else:
                merged[name] = value
        return merged

    @staticmethod
    def correction(state: dict[str, Tensor], stem: str, dtype: torch.dtype) -> Float[Tensor, "out in"]:
        """The rank-r term one adapted layer learned, at the scale `LoRALinear.forward` applies it."""
        down = state[f"{stem}{DOWN}"]
        up = state[f"{stem}{UP}"]
        return ((up @ down) * (ADAPTER_ALPHA / down.shape[0])).to(dtype)

    def write(self, destination: Path) -> Path:
        """Write the file `PretrainedEncoder` reads, and say what it came from.

        `config` records the source rather than the objective's knobs, which a member has none of. It
        is the same question the pretrainer's config answers — which run produced these weights — and
        the write-up asks it first.
        """
        state = self.backbone_state()
        torch.save(
            {
                "encoder": state,
                "variant": self.variant(),
                "cache": self.cache(),
                "config": {"from_member": self.path.name, "holdout": self.holdout(), "merged_lora": True},
            },
            destination,
        )
        logger.info(
            "wrote %s: %d tensors from %s (holdout %.4f, variant %s)",
            destination,
            len(state),
            self.path.name,
            self.holdout(),
            self.variant(),
        )
        return destination


def main() -> None:
    """Convert one fitted member into an encoder file, so `--encoder-weights` can start from it."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("member", type=Path, help="a fitted member .pt")
    command.parser.add_argument("destination", type=Path, help="where to write the encoder file")
    parsed = command.parsed()
    FittedEncoder(parsed.member).write(parsed.destination)


if __name__ == "__main__":
    main()
