"""The mirror for the 2048-wide volumetric trunk, run against a stand-in ResNet rather than 175 MB.

The stand-in is the point, not a shortcut: what this file has to pin is the SHAPE contract and the
intensity preparation, and both are the trunk's own code. A test that loaded their checkpoint would be
measuring their network and would need a data root, which a unit test may not touch.
"""

from typing import override

import torch
from torch import Tensor, nn

from core.data.corpus.slots import SLOTS
from kneemri.models.medicalnet import DEPTH, READ_SIZE, WIDTH, MedicalNetTrunk

SLICES = 4
"""Fewer than the cache's twelve, so the depth resample has something to do."""

EDGE = 8
"""A tiny frame — the trunk resamples to `READ_SIZE` regardless, which is what makes it tiny enough."""

BRIGHT = 200
"""The high end of the synthetic stack, so the z-score cannot be a no-op."""

STRIDE = 8
"""How far the stand-in's single pooling collapses the spatial axes, standing in for their conv1,
maxpool and layer2 strides. Only that the axes SURVIVE to `pooled` matters here."""


class _Stage(nn.Module):
    """One of their named stages, as an identity — the trunk walks them by name, so they must exist."""

    @override
    def forward(self, activation: Tensor) -> Tensor:
        """Hand the activation on untouched."""
        return activation


class _Widen(nn.Module):
    """The stage that sets the feature width, standing in for `layer4`'s 512x4 bottleneck output."""

    @override
    def forward(self, activation: Tensor) -> Tensor:
        """Collapse the spatial axes cheaply and expand to the real trunk's width."""
        pooled = nn.functional.avg_pool3d(activation, STRIDE)
        return pooled.mean(dim=1, keepdim=True).expand(-1, WIDTH, -1, -1, -1)


class _ResNet(nn.Module):
    """A stand-in carrying their attribute NAMES, which is the contract `encoded` actually depends on."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = _Stage()
        self.bn1 = _Stage()
        self.relu = _Stage()
        self.maxpool = _Stage()
        self.layer1 = _Stage()
        self.layer2 = _Stage()
        self.layer3 = _Stage()
        self.layer4 = _Widen()


def _pixels() -> Tensor:
    """One batch of the pixel cache's own dtype and rank: studies, slots, slices, height, width."""
    pixels = torch.zeros((2, len(SLOTS), SLICES, EDGE, EDGE), dtype=torch.uint8)
    pixels[:, :, SLICES - 1] = BRIGHT
    return pixels


def test_prepared() -> None:
    """One slot becomes a single-channel volume at their depth and size, z-scored over its own tissue."""
    prepared = MedicalNetTrunk.prepared(_pixels()[:, 0])
    assert prepared.shape == (2, 1, DEPTH, READ_SIZE, READ_SIZE)
    assert float(prepared.max()) > 0.0
    assert float(prepared.std()) > 0.0


def test_prepared_holds_a_missing_slot_finite() -> None:
    """An unacquired slot is all zeros in the cache and has no non-zero voxels to z-score at all."""
    blank = torch.zeros((1, SLICES, EDGE, EDGE), dtype=torch.uint8)
    prepared = MedicalNetTrunk.prepared(blank)
    assert bool(torch.isfinite(prepared).all())
    assert float(prepared.abs().max()) == 0.0


def test_pooled() -> None:
    """The spatial axes collapse and the channel axis survives, which is the feature width."""
    activation = torch.ones((2, WIDTH, 2, EDGE, EDGE))
    pooled = MedicalNetTrunk.pooled(activation)
    assert pooled.shape == (2, WIDTH)
    assert float(pooled[0, 0]) == 1.0


def test_encoded() -> None:
    """One slot through the trunk gives one vector per study of `layer4`'s width."""
    assert MedicalNetTrunk(_ResNet()).encoded(_pixels()[:, 0]).shape == (2, WIDTH)


def test_slice_features() -> None:
    """The whole-stack contract: a slot yields ONE vector, so the depth axis of the cache is 1."""
    features = MedicalNetTrunk(_ResNet()).slice_features(_pixels())
    assert features.shape == (2, len(SLOTS), 1, WIDTH)
