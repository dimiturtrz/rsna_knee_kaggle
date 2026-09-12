"""Their published pretraining, loaded into a network this process built — the fetch half of the trunk.

Split from `medicalnet` for the reason `ortho_weights` is split from `ortho_diffusion`: that file says
what a feature IS, this one says where the parameters come from and refuses a checkpoint that does not
fit. A loader that quietly falls back is the failure mode every foreign-weight bug in this project has
taken.

    git clone https://github.com/Tencent/MedicalNet external/MedicalNet   # pinned 20f76aaa
    # their Google-Drive/Kaggle mirror of `pretrain/` -> D:/data/external/medicalnet
"""

import importlib.util
import logging
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from torch import nn

from kneemri.models.medicalnet import (
    CHECKPOINT,
    DEPTH,
    EXTERNAL,
    LAST_STAGE,
    READ_SIZE,
    SEGMENTATION_CLASSES,
    SHORTCUT,
    MedicalNetTrunk,
)

logger = logging.getLogger(__name__)

MODULE = "models/resnet.py"
"""Their network definition inside the checkout — the one file of theirs this project ever executes."""

PARALLEL_PREFIX = "module."
"""Their checkpoint is saved from a `DataParallel` wrapper, so every tensor key carries it."""

STATE = "state_dict"
"""The key their `train.py` saves the weights under, beside an `epoch` and an optimiser state."""

HEAD = "conv_seg"
"""The segmentation head their constructor builds and their pretraining checkpoint does NOT carry. It is
the ONLY thing allowed to be missing — the trunk never runs it, and `loaded` names it rather than
passing `strict=False` and accepting whatever else happens to be absent."""


class MedicalNetWeights:
    """Their checkpoint on disk, and the network that accepts it or does not load at all."""

    def __init__(self, directory: Path, checkout: Path = EXTERNAL) -> None:
        self.directory = directory
        self.checkout = checkout

    def factory(self) -> Callable[..., nn.Module]:  # devtools-ignore: test-mirror
        """Their `resnet50`, loaded from the checkout BY FILE PATH rather than vendored.

        `spec_from_file_location` where `ortho_weights` needs `sys.path`, and the difference is a fact
        about their code rather than a preference: OrthoDiffusion's `unet.py` imports its package
        siblings, which only a real package import resolves, while this file imports nothing but torch.
        The narrower route is the right one — putting their checkout on `sys.path` would publish a
        top-level `models` package into a process that already has `kneemri.models`.
        """
        source = self.checkout.resolve() / MODULE
        spec = importlib.util.spec_from_file_location("medicalnet_resnet", source)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"no MedicalNet checkout at {source} — see this module's docstring")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return cast("Callable[..., nn.Module]", module.resnet50)

    @staticmethod
    def stripped(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The network's own parameters, with the `DataParallel` prefix removed.

        A checkpoint saved from the bare network has no prefix and nothing to strip, which would leave an
        empty mapping — so an empty result means "not wrapped", never "no parameters", and the caller
        falls back to the whole state rather than loading nothing.
        """
        return {
            key.removeprefix(PARALLEL_PREFIX): value
            for key, value in weights.items()
            if key.startswith(PARALLEL_PREFIX)
        }

    def network(self) -> nn.Module:  # devtools-ignore: test-mirror
        """Their ResNet-50 at this probe's input geometry, before any weights are in it."""
        return self.factory()(
            sample_input_D=DEPTH,
            sample_input_H=READ_SIZE,
            sample_input_W=READ_SIZE,
            num_seg_classes=SEGMENTATION_CLASSES,
            shortcut_type=SHORTCUT,
        )

    def trunk(self, device: torch.device, stage: str = LAST_STAGE) -> MedicalNetTrunk:  # devtools-ignore: test-mirror
        """The encoder `build_medicalnet_embeddings` hands to `EmbeddingCache`, frozen and in eval mode.

        `weights_only=True` because a foreign `.pth` is an untrusted pickle, and the refusal on any
        unexpected key — or on any missing key outside the segmentation head — because the alternative
        accepts a mismatch and hands back a partly random network that encodes perfectly happily, at full
        speed, into a cache that looks exactly like a real one.
        """
        state = torch.load(self.directory / CHECKPOINT, map_location="cpu", weights_only=True)
        weights = cast("dict[str, torch.Tensor]", state.get(STATE, state))
        network = self.network()
        report = network.load_state_dict(self.stripped(weights) or weights, strict=False)
        unexpected = list(report.unexpected_keys)
        missing = [key for key in report.missing_keys if not key.startswith(f"{HEAD}.")]
        if unexpected or missing:
            raise ValueError(f"checkpoint does not fit: unexpected={unexpected[:4]} missing={missing[:4]}")
        logger.info("loaded %s from %s (%d tensors)", CHECKPOINT, self.directory, len(weights))
        frozen = network.to(device).eval().requires_grad_(requires_grad=False)
        return MedicalNetTrunk(frozen, stage).to(device)
