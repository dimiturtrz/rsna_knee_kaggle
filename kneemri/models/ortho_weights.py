"""Their published checkpoints, loaded into networks this process built — the fetch half of the trunk.

Split from `ortho_diffusion` because the two are different subjects: that file says what a feature IS,
this one says where the parameters come from and refuses a checkpoint that does not fit. The gate that
insisted on the split is right about why — a loader that quietly falls back is the failure mode every
foreign-weight bug in this project has taken.

    git clone https://github.com/lt-0123/OrthoDiffusion external/OrthoDiffusion   # pinned 99d1518, MIT
    huggingface-cli download <card> --local-dir D:/data/external/orthodiffusion   # three 553 MB files
"""

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from torch import nn

from kneemri.models.ortho_diffusion import (
    CHANNELS,
    EXTERNAL,
    PLANE_READS,
    READ_SIZE,
    RESIDUAL_BLOCKS,
    OrthoDiffusionTrunk,
)

logger = logging.getLogger(__name__)

EMA = "ema"
"""The averaged weights inside their checkpoint, preferred over the raw `model` for a frozen read."""

RAW = "model"
"""The un-averaged fallback, for a checkpoint saved without an EMA copy."""

DENOISER_PREFIX = "denoise_fn."
"""Their checkpoint is saved from the `GaussianDiffusion` wrapper, so every U-Net key is prefixed. The
wrapper's own schedule buffers are dropped: `diffusion_schedule` rebuilds them from the same arithmetic."""


class DenoiserWeights:
    """Their checkpoints on disk, and the networks that accept them or refuse to load at all."""

    def __init__(self, directory: Path, checkout: Path = EXTERNAL) -> None:
        self.directory = directory
        self.checkout = checkout

    def factory(self) -> Callable[..., nn.Module]:  # devtools-ignore: test-mirror
        """Their `create_model`, imported from the checkout by path.

        `sys.path` rather than `spec_from_file_location` — the route `foreign_member` takes for raptor —
        because `unet.py` uses package-relative imports (`from ._utils import ...`) that only resolve
        when its package is importable. Their single flat script had no such requirement.
        """
        root = str(self.checkout.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from diffusion_model.unet import create_model  # noqa: PLC0415  # pyrefly: ignore[missing-import]

        return cast("Callable[..., nn.Module]", create_model)

    @staticmethod
    def stripped(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The U-Net's own parameters, with the diffusion wrapper's prefix and buffers removed.

        A checkpoint saved from the bare network has no prefix and nothing to strip, which would leave an
        empty mapping — so an empty result means "not prefixed", never "no parameters", and the caller
        falls back to the whole state rather than loading nothing.
        """
        return {
            key.removeprefix(DENOISER_PREFIX): value
            for key, value in weights.items()
            if key.startswith(DENOISER_PREFIX)
        }

    def loaded(self, plane: str, device: torch.device) -> nn.Module:  # devtools-ignore: test-mirror
        """One plane's U-Net with their averaged weights in it, frozen and in eval mode.

        `strict=True` on purpose: the alternative accepts a checkpoint whose keys do not match and hands
        back a randomly-initialised network that encodes perfectly happily, at full speed, into a cache
        that looks exactly like a real one.
        """
        network = self.factory()(READ_SIZE, CHANNELS, RESIDUAL_BLOCKS, in_channels=1, out_channels=1)
        state = torch.load(self.directory / PLANE_READS[plane].checkpoint, map_location="cpu", weights_only=False)
        weights = cast("dict[str, torch.Tensor]", state[EMA] if EMA in state else state[RAW])
        network.load_state_dict(self.stripped(weights) or weights, strict=True)
        return network.to(device).eval().requires_grad_(requires_grad=False)

    def trunk(self, device: torch.device) -> OrthoDiffusionTrunk:  # devtools-ignore: test-mirror
        """All three planes, as the encoder `build_ortho_embeddings` hands to `EmbeddingCache`."""
        denoisers = {plane: self.loaded(plane, device) for plane in PLANE_READS}
        logger.info("loaded %d plane denoisers from %s", len(denoisers), self.directory)
        return OrthoDiffusionTrunk(denoisers).to(device)
