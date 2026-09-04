"""A tiny published archive, rebuilt and read back — every shape and every permutation known in advance."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.findings import TARGETS
from kneemri.inference.foreign_member import (
    CHANNELS,
    ForeignMember,
    ForeignSpec,
    PublishedArchive,
    WindowFeed,
)

ARCH = "resnet18"
"""A timm architecture small enough for a unit test and built with `pretrained=False`, so no network."""

DIM = 512
"""`resnet18`'s pooled feature width — the head's input, and what a checkpoint's `clsW` would declare."""

RES = 32
SLOTS = 2
SLICES = 4
SIZE = 16
STUDIES = 3
GOLD = 0.9167


def _labels() -> list[str]:
    """Their order, which is ours rotated — a permutation the reader must undo and cannot ignore."""
    return [*TARGETS[1:], TARGETS[0]]


def _payload() -> PublishedArchive:
    """One archive in the published layout: weights under `model`, the spec beside them."""
    member = ForeignMember(ForeignSpec(ARCH, RES, tuple(_labels()), GOLD, DIM))
    weights = {f"backbone.{name}": value for name, value in member.backbone.state_dict().items()}
    weights.update({name: value for name, value in member.head.state_dict().items() if not name.startswith("att.2")})
    weights["att.3.weight"] = member.head.att[2].weight
    weights["att.3.bias"] = member.head.att[2].bias
    return {"model": weights, "arch": ARCH, "res": RES, "lab": _labels(), "gold_auc": GOLD}


def _feed(path: Path) -> WindowFeed:
    """A cache of three studies whose pixels are their own row index, so a misread row is visible."""
    path.mkdir(parents=True, exist_ok=True)
    images = np.zeros((STUDIES, SLOTS, SLICES, SIZE, SIZE), dtype=np.uint8)
    for row in range(STUDIES):
        images[row] = row + 1
    np.save(path / "images_k.npy", images)
    np.save(path / "mask_k.npy", np.ones((STUDIES, SLOTS), dtype=np.float32))
    (path / "index_k.json").write_text(json.dumps([f"s{row}" for row in range(STUDIES)]), encoding="utf-8")
    return WindowFeed.of(path, "k", RES)


def test_foreign_spec_of():
    """Every field of the spec comes off the archive, including the width its own `clsW` declares."""
    spec = ForeignSpec.of(_payload())

    assert spec.arch == ARCH
    assert spec.res == RES
    assert spec.dim == DIM
    assert spec.gold == pytest.approx(GOLD)
    assert spec.labels == tuple(_labels())


def test_foreign_spec_order():
    """The permutation puts their label order into ours, and is a permutation of every index."""
    order = ForeignSpec(ARCH, RES, tuple(_labels()), GOLD, DIM).order

    assert sorted(order.tolist()) == list(range(len(TARGETS)))
    assert [_labels()[index] for index in order] == list(TARGETS), "reordered rows must read as TARGETS"


def test_rebuild(tmp_path: Path):
    """A saved archive comes back as a member whose outputs are already in our label order."""
    torch.save(_payload(), tmp_path / "foreign.pt")

    member = ForeignMember.rebuild(tmp_path / "foreign.pt")

    assert member.spec.arch == ARCH
    logits = member(torch.zeros(1, 2, CHANNELS, RES, RES))
    assert logits.shape == (1, len(TARGETS))
    expected = torch.from_numpy(ForeignSpec.of(_payload()).order)
    assert torch.equal(member.get_buffer("permutation"), expected)


def test_windows(tmp_path: Path):
    """Every slot contributes every consecutive triple, and a missing slot contributes none."""
    feed = _feed(tmp_path / "cache")

    bags = feed.windows(1)

    assert bags.shape == (SLOTS * (SLICES - CHANNELS + 1), CHANNELS, SIZE, SIZE)
    assert (bags == 2).all(), "row 1 holds twos; any other value means the wrong study was read"

    feed.mask[1, 0] = 0.0
    assert feed.windows(1).shape[0] == SLICES - CHANNELS + 1, "a slot the decoder never filled is dropped"


def test_batch(tmp_path: Path):
    """Windows reach the member at its own read size, scaled into [0,1] and not normalised further."""
    feed = _feed(tmp_path / "cache")

    pixels = feed.batch(2, torch.device("cpu"))

    assert pixels.shape[-2:] == (RES, RES)
    assert pixels.min() >= 0.0 and pixels.max() <= 1.0
    assert pixels.mean().item() == pytest.approx(3.0 / 255.0, abs=1e-3), "three, scaled, and nothing subtracted"


def test_window_feed_of(tmp_path: Path):
    """The cache and its per-slot presence mask are read from the two arrays that name the key."""
    feed = _feed(tmp_path / "cache")

    assert feed.images.shape == (STUDIES, SLOTS, SLICES, SIZE, SIZE)
    assert feed.mask.shape == (STUDIES, SLOTS)
    assert feed.res == RES


def test_window_feed_order(tmp_path: Path):
    """The study order the cache's rows are indexed by, which is what a holdout is joined on."""
    _feed(tmp_path / "cache")

    assert WindowFeed.order(tmp_path / "cache", "k") == [f"s{row}" for row in range(STUDIES)]


def test_forward():
    """Every window is encoded and the twelve logits come back already permuted into our order."""
    member = ForeignMember(ForeignSpec(ARCH, RES, tuple(_labels()), GOLD, DIM)).eval()

    logits = member.forward(torch.zeros(1, 2, CHANNELS, RES, RES))

    assert logits.shape == (1, len(TARGETS))
    assert torch.isfinite(logits).all()
