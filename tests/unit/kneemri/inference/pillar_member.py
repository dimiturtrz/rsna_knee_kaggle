"""A stubbed 3D trunk, rebuilt and read back — the saturation bug is the assertion that matters here.

The trunk class is imported from a third-party checkout this repo does not vendor, so every test below
substitutes `AtlasImport.atlas_class` with a module whose state dict has the same NAMES and none of the depth. That
keeps the reviewable half — the spec read off a checkpoint, the feed's slot handling, and the
normalisation the head is fitted against — checkable on a machine that has never run `git clone`.
"""

import json
from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from core.findings import TARGETS
from kneemri.inference import pillar_member
from kneemri.inference.pillar_member import (
    CHANNELS,
    MODALITY,
    AtlasConfig,
    PillarMember,
    PillarSpec,
    VolumeFeed,
)

SCALES = 3
"""Scales the stub config declares, so `dim` divides by it the way the shipped config's does."""

DIM = SCALES * 4
SLOTS = 2
DEPTH = 4
SIZE = 8
STUDIES = 3
GRID = (2, 4, 4)
"""A grid small enough to interpolate in a unit test. The real one is 192x384x384, which is 340 MB a series."""

EMBED = "model.visual.patch_embeds.breast_mr.conv_down.0.weight"


class _Conv(nn.Module):
    """One named weight, so a checkpoint key can land on something with the publisher's own name."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, CHANNELS, 1, 1, 1))


class _Trunk(nn.Module):
    """A stand-in for `MultiModalAtlas`: same constructor keywords, same key names, no depth.

    Its features are deliberately UNNORMALISED and large, because that is exactly the state the real
    trunk returns and the state `PillarMember.embedding` exists to correct.
    """

    def __init__(self, *, embed_dim: int, model_config: AtlasConfig, **rest: object) -> None:
        super().__init__()
        del rest
        self.width = embed_dim * int(model_config["model"]["atlas_args"]["num_scales"])
        self.patch_embeds = nn.ModuleDict({MODALITY: nn.Module()})
        self.patch_embeds[MODALITY].conv_down = nn.ModuleList([_Conv()])

    @override
    def forward(self, volumes: dict[str, Tensor]) -> Tensor:
        """A constant, large, positive feature vector — one row, whatever the volume was."""
        batch = volumes[MODALITY].shape[0]
        return torch.full((batch, self.width), 5.0)


def _config(path: Path) -> Path:
    """The publisher's config trimmed to the two branches `PillarSpec.of` reads."""
    modality = {"image_size": [0, 0, 0], "in_chans": 0, "patch_size": [6, 6, 6]}
    body = {
        "vision_cfg": {
            "model_config": {"model": {"atlas_args": {"num_scales": SCALES}}, "modalities": {"ct": modality}}
        }
    }
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _payload() -> dict[str, dict[str, Tensor]]:
    """One archive in the published layout: a CLIP pair, of which only the visual half is ever loaded."""
    return {
        "model": {EMBED: torch.ones(2, CHANNELS, 1, 1, 1), "model.text.proj.0.weight": torch.zeros(2, 2)},
        "head": {"weight": torch.zeros(len(TARGETS), DIM), "bias": torch.zeros(len(TARGETS))},
    }


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`atlas_class` answers with the stub, and `CONFIG` with a trimmed copy of the publisher's."""
    monkeypatch.setattr(pillar_member.AtlasImport, "atlas_class", lambda: _Trunk)
    monkeypatch.setattr(pillar_member, "CONFIG", _config(tmp_path / "config.json"))


def _member() -> PillarMember:
    """A member over the stub trunk, with a head whose weights are known and non-zero."""
    spec = PillarSpec.of(_payload())
    member = PillarMember(spec)
    with torch.no_grad():
        member.head.weight.fill_(0.25)
        member.head.bias.zero_()
    return member.eval()


def _feed(path: Path, grid: tuple[int, int, int] = GRID) -> VolumeFeed:
    """A cache of three studies whose voxels vary within a series, so a z-score is defined."""
    path.mkdir(parents=True, exist_ok=True)
    ramp = np.arange(DEPTH * SIZE * SIZE, dtype=np.uint8).reshape(DEPTH, SIZE, SIZE)
    images = np.stack([np.stack([ramp + row for _ in range(SLOTS)]) for row in range(STUDIES)])
    np.save(path / "images_k.npy", images)
    np.save(path / "mask_k.npy", np.ones((STUDIES, SLOTS), dtype=np.float32))
    (path / "index_k.json").write_text(json.dumps([f"s{row}" for row in range(STUDIES)]), encoding="utf-8")
    feed = VolumeFeed.of(path, "k")
    return VolumeFeed(feed.images, feed.mask, grid)


def test_pillar_spec_of(stubbed: None):
    """The modality block is rebuilt from the checkpoint's own patch-embed, not from the shipped config."""
    del stubbed

    spec = PillarSpec.of(_payload())

    assert spec.dim == DIM
    assert spec.labels == TARGETS
    block = spec.model_config["modalities"][MODALITY]
    assert block["in_chans"] == CHANNELS, "in_chans is read off the convolution, never assumed"
    assert block["image_size"] == list(pillar_member.LAYOUT), "the declared grid is the token grid, not the volume"


def test_pillar_spec_order(stubbed: None):
    """The permutation is a permutation of every index, and reorders their labels into ours."""
    del stubbed

    order = PillarSpec.of(_payload()).order

    assert sorted(order.tolist()) == list(range(len(TARGETS)))
    assert [pillar_member.PILLAR_LABELS[index] for index in order] == list(TARGETS)


def test_embedding(stubbed: None):
    """Trunk features reach the head L2-normalised, which is the whole difference between a reading and ties."""
    del stubbed
    member = _member()

    features = member.embedding(torch.zeros(CHANNELS, *GRID))

    assert features.shape == (1, DIM)
    assert features.norm(dim=-1).item() == pytest.approx(1.0), "an unnormalised feature saturates every sigmoid"
    assert torch.sigmoid(member.head(features)).max().item() < 1.0, "no probability may land on exactly one"


def test_forward(stubbed: None):
    """Every series is read and the twelve log-odds come back finite, in our order."""
    del stubbed
    member = _member()

    logits = member.forward(torch.zeros(1, SLOTS, CHANNELS, *GRID))

    assert logits.shape == (1, len(TARGETS))
    assert torch.isfinite(logits).all(), "the log-odds round-trip must survive a saturated probability"


def test_rebuild(stubbed: None, tmp_path: Path):
    """A saved archive comes back as a member; its text tower is dropped rather than loaded."""
    del stubbed
    torch.save(_payload(), tmp_path / "pillar.pt")

    member = PillarMember.rebuild(tmp_path / "pillar.pt")

    assert member.spec.dim == DIM
    assert not member.training, "a member that rebuilds in train mode reads with dropout on"
    weight = member.trunk.state_dict()[f"patch_embeds.{MODALITY}.conv_down.0.weight"]
    assert torch.equal(weight, torch.ones(2, CHANNELS, 1, 1, 1))


def test_volume_feed_of(tmp_path: Path):
    """The cache and its per-slot presence mask are read from the two arrays that name the key."""
    feed = _feed(tmp_path / "cache")

    assert feed.images.shape == (STUDIES, SLOTS, DEPTH, SIZE, SIZE)
    assert feed.mask.shape == (STUDIES, SLOTS)


def test_volume_feed_order(tmp_path: Path):
    """The study order the cache's rows are indexed by, which is what a holdout is joined on."""
    _feed(tmp_path / "cache")

    assert VolumeFeed.order(tmp_path / "cache", "k") == [f"s{row}" for row in range(STUDIES)]


def test_volumes(tmp_path: Path):
    """Every present slot is one series, and a slot the decoder never filled is dropped."""
    feed = _feed(tmp_path / "cache")

    assert feed.volumes(1).shape == (SLOTS, DEPTH, SIZE, SIZE)

    feed.mask[1, 0] = 0.0
    assert feed.volumes(1).shape[0] == SLOTS - 1, "an empty volume is a confident prediction of nothing"


def test_batch(tmp_path: Path):
    """Series are z-scored per series, resampled to the grid, then replicated across three channels."""
    feed = _feed(tmp_path / "cache")

    pixels = feed.batch(2, torch.device("cpu"))

    assert pixels.shape == (SLOTS, CHANNELS, *GRID)
    assert torch.isfinite(pixels).all()
    assert torch.equal(pixels[:, 0], pixels[:, 2]), "the channel axis is a replicate, and carries no depth"
    assert pixels[0, 0].mean().item() == pytest.approx(0.0, abs=0.1), "z-scored before the resample, not after"
