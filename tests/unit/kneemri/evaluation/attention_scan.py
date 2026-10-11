from typing import cast, override

import numpy as np
import torch
from torch import nn

from core.data.corpus.slots import SLOTS
from core.findings import TARGETS
from kneemri.evaluation.attention_scan import AttentionScan
from kneemri.inference.holdout_predictions import ServedModel
from kneemri.inference.holdout_slab import HoldoutSlab
from kneemri.inference.members import Window
from kneemri.models.model import StudyModel

SEED = 2026
STUDIES = 8
SLICES = 4
SIZE = 8
DIM = 4
CARRIER = 0
"""The slot the fake head puts all its attention on, so a scan has a knowable right answer."""


class _SlotAttender(nn.Module):
    """A stand-in whose attention is entirely on one slot — the degenerate read, deliberately."""

    def slot_features(self, imgs: torch.Tensor, img_size: int | None = None) -> torch.Tensor:
        del img_size
        return imgs.float().mean(dim=(2, 3, 4)).unsqueeze(-1).expand(-1, -1, DIM)

    def slot_weights(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        del mask
        weights = torch.zeros(x.shape[0], len(TARGETS), x.shape[1])
        weights[:, :, CARRIER] = 1.0
        return weights

    @property
    def head(self) -> "_SlotAttender":
        return self

    @override
    def forward(self, imgs: torch.Tensor, mask: torch.Tensor, img_size: int | None = None) -> torch.Tensor:
        del img_size
        return self.slot_features(imgs).mean(-1) * mask


def test_weights() -> None:
    """One row per study, per finding, per slot, meaned over the member's own window starts."""
    generator = np.random.default_rng(SEED)
    images = generator.integers(0, 255, (STUDIES, len(SLOTS), SLICES, SIZE, SIZE), dtype=np.uint8)
    mask = np.ones((STUDIES, len(SLOTS)), dtype=np.float32)
    model = cast(StudyModel, _SlotAttender())
    window = Window(starts=[0, 1], width=2, size=SIZE)

    weights = AttentionScan(torch.device("cpu"), batch=4).weights(
        ServedModel(model, window), HoldoutSlab.of(images, mask)
    )

    assert weights.shape == (STUDIES, len(TARGETS), len(SLOTS))
    assert np.allclose(weights.sum(axis=2), 1.0), "a mean of softmaxes is still a distribution"
    assert np.allclose(weights[:, :, CARRIER], 1.0)
