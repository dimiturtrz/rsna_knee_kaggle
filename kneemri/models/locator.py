"""A per-slice tissue locator: a pretrained ViT's multi-depth patch features under a light upsampling head.

Five classes, not SKM-TEA's seven: `MERGED` folds the medial and lateral tibial cartilage together and
the two menisci together, because a single sagittal slice does not show which side of the knee it is
on — the side is a property of the slice's POSITION in the stack, and it is recovered there.

Features are read at three depths (`LAYERS`) because the first thing the head has to do is draw a thin
boundary, which the last layer's semantics alone carry poorly at a 24 x 24 token grid. The head
upsamples twice by two and the logits are resized the rest of the way.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast, override

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, UInt8
from torch import Tensor, nn

from core.data.corpus.skmtea_slices import IMAGES, LABELS, SLICE_TABLE, SPLIT_COLUMN
from kneemri.models.encoder import Encoder

CLASSES = ("background", "patellar cartilage", "femoral cartilage", "tibial cartilage", "meniscus")
MERGED = (0, 1, 2, 3, 3, 4, 4)
"""SKM-TEA's mask value (0 background, then its six tissues) -> the locator's class."""
LAYERS = (4, 8, 12)
HIDDEN = (256, 128, 64)
PATCH = 14
BYTE = 255.0
EVAL_BATCH = 32


class Locator(nn.Module):
    """Slices in, per-pixel tissue logits out, at the slices' own size."""

    mean: Tensor
    std: Tensor

    def __init__(self, trunk: nn.Module, encoder: Encoder, layers: Sequence[int] = LAYERS) -> None:
        super().__init__()
        self.trunk = trunk
        self.layers = tuple(layers)
        self.dim = encoder.dim
        wide, middle, narrow = HIDDEN
        self.head = nn.Sequential(
            nn.Conv2d(encoder.dim * len(self.layers), wide, 1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(wide, middle, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(middle, narrow, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(narrow, len(CLASSES), 1),
        )
        self.register_buffer("mean", torch.tensor(encoder.normalization.mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(encoder.normalization.std).view(1, 3, 1, 1))

    @staticmethod
    def pretrained(encoder: Encoder) -> "Locator":
        return Locator(encoder.load(), encoder)

    @staticmethod
    def merged(labels: Int[Tensor, "*shape"]) -> Int[Tensor, "*shape"]:
        return torch.tensor(MERGED, device=labels.device)[labels.long()]

    @override
    def forward(self, slices: Float[Tensor, "n h w"] | UInt8[Tensor, "n h w"]) -> Float[Tensor, "n classes h w"]:
        """Byte-scaled grey slices (0-255) -> logits. The side must tile by the trunk's patch."""
        batch, height, width = slices.shape
        pixels = ((slices.float() / BYTE)[:, None].expand(-1, 3, -1, -1) - self.mean) / self.std
        hidden = self.trunk(pixel_values=pixels, output_hidden_states=True).hidden_states
        grid = (height // PATCH, width // PATCH)
        features = torch.cat(
            [cast("Tensor", hidden[layer])[:, 1:].mT.reshape(batch, self.dim, *grid) for layer in self.layers],
            dim=1,
        )
        return F.interpolate(self.head(features), size=(height, width), mode="bilinear")

    def groups(self, trunk_rate: float, head_rate: float) -> list[dict[str, float | list[nn.Parameter]]]:
        """Optimizer parameter groups: the pretrained trunk at its own, slower rate."""
        return [
            {"params": list(self.trunk.parameters()), "lr": trunk_rate},
            {"params": list(self.head.parameters()), "lr": head_rate},
        ]

    def predict(self, slices: Float[Tensor, "n h w"] | UInt8[Tensor, "n h w"]) -> Int[Tensor, "n h w"]:
        with torch.no_grad():
            return self(slices).argmax(1)


@dataclass(frozen=True)
class SliceCorpus:
    """The slice arrays and the split each slice's scan belongs to."""

    images: UInt8[np.ndarray, "slice echo h w"]
    masks: UInt8[np.ndarray, "slice h w"]
    splits: pd.Series

    @staticmethod
    def read(directory: Path) -> "SliceCorpus":
        return SliceCorpus(
            np.load(directory / IMAGES, mmap_mode="r"),
            np.load(directory / LABELS, mmap_mode="r"),
            pd.read_csv(directory / SLICE_TABLE)[SPLIT_COLUMN],
        )

    def indices(self, split: str) -> Int[np.ndarray, " k"]:
        return np.flatnonzero(self.splits.to_numpy() == split)

    def batch(
        self, indices: Int[np.ndarray, " k"], echoes: Int[np.ndarray, " k"], device: torch.device
    ) -> tuple[UInt8[Tensor, "k h w"], Int[Tensor, "k h w"]]:
        order = np.argsort(indices)
        chosen, echo = indices[order], echoes[order]
        slices = torch.from_numpy(np.stack([self.images[i, e] for i, e in zip(chosen, echo, strict=True)]))
        labels = torch.from_numpy(self.masks[chosen].astype(np.int64))
        return slices.to(device), Locator.merged(labels.to(device))
