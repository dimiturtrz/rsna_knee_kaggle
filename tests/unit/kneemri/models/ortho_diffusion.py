"""The mirror for the volumetric trunk, run against a stand-in denoiser rather than 1.6 GB of weights.

The stand-in is the point, not a shortcut: what this file has to pin is the SHAPE contract and the
intensity preparation, and both are the trunk's own code. A test that loaded their checkpoints would be
measuring their network and would need a data root, which a unit test may not touch.
"""

from typing import override

import torch
from torch import Tensor, nn

from core.data.corpus.slots import SLOTS, Plane
from kneemri.models.ortho_diffusion import DEPTH, MIDDLE_WIDTH, PLANE_READS, READ_SIZE, OrthoDiffusionTrunk

SLICES = 4
"""Fewer than the cache's twelve, so the depth resample has something to do."""

EDGE = 8
"""A tiny frame — the trunk resamples to `READ_SIZE` regardless, which is what makes it tiny enough."""

BRIGHT = 200
"""The high end of the synthetic stack, so min-max normalisation cannot be a no-op."""


class _Denoiser(nn.Module):
    """A stand-in with their forward's signature: the middle activations, keyed as their U-Net keys them."""

    @override
    def forward(
        self, noisy: Tensor, steps: Tensor, *, ret_mid: bool = False
    ) -> tuple[None, dict[str, dict[str, Tensor]]]:
        """Return one middle activation of the real trunk's width, ignoring the timestep."""
        del steps, ret_mid
        pooled = noisy.mean(dim=1, keepdim=True).expand(-1, MIDDLE_WIDTH, -1, -1, -1)
        return None, {"mid": {read.block: pooled for read in PLANE_READS.values()}}


def _trunk() -> OrthoDiffusionTrunk:
    """A trunk whose three planes are stand-ins, built entirely inside this test."""
    return OrthoDiffusionTrunk({plane: _Denoiser() for plane in PLANE_READS})


def _pixels() -> Tensor:
    """One batch of the pixel cache's own dtype and rank: studies, slots, slices, height, width."""
    pixels = torch.zeros((2, len(SLOTS), SLICES, EDGE, EDGE), dtype=torch.uint8)
    pixels[:, :, SLICES - 1] = BRIGHT
    return pixels


def test_prepared() -> None:
    """One slot becomes a single-channel volume at their depth and size, scaled into [-1, 1]."""
    prepared = OrthoDiffusionTrunk.prepared(_pixels()[:, 0])
    assert prepared.shape == (2, 1, DEPTH, READ_SIZE, READ_SIZE)
    assert float(prepared.min()) == -1.0
    assert float(prepared.max()) == 1.0


def test_prepared_holds_a_missing_slot_finite() -> None:
    """An unacquired slot is all zeros in the cache and must not divide by zero here."""
    blank = torch.zeros((1, SLICES, EDGE, EDGE), dtype=torch.uint8)
    assert bool(torch.isfinite(OrthoDiffusionTrunk.prepared(blank)).all())


def test_pooled() -> None:
    """The spatial axes collapse and the channel axis survives, which is the feature width."""
    activation = torch.ones((2, MIDDLE_WIDTH, DEPTH, EDGE, EDGE))
    pooled = OrthoDiffusionTrunk.pooled(activation)
    assert pooled.shape == (2, MIDDLE_WIDTH)
    assert float(pooled[0, 0]) == 1.0


def test_encoded() -> None:
    """One slot through its plane's denoiser gives one vector per study of the middle block's width."""
    encoded = _trunk().encoded(_pixels()[:, 0], Plane.SAGITTAL)
    assert encoded.shape == (2, MIDDLE_WIDTH)


def test_slice_features() -> None:
    """The whole-stack contract: a slot yields ONE vector, so the depth axis of the cache is 1."""
    features = _trunk().slice_features(_pixels())
    assert features.shape == (2, len(SLOTS), 1, MIDDLE_WIDTH)
