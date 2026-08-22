"""The encoded corpus: what its name states, and that what it stores is what a head will read."""

from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from torch import Tensor

from core.data.corpus.study_cache import CachedCorpus
from kneemri.embeddings.embedding_cache import EmbeddedCorpus, EmbeddingCache, EmbeddingSpec

STUDIES = 4
SLOTS = 3
SLICES = 2
WIDTH = 5
PIXELS = 8

SPEC = EmbeddingSpec(variant="small", source=f"336px_{SLICES}sl", read_size=336, pool="cls_mean", slice_read="max")


class _FakeEncoder(torch.nn.Module):
    """Stands in for `StudyModel`, returning features that DEPEND on the pixels it was given.

    Constant features would let a cache that stores the wrong studies, or stores them in the wrong
    order, pass every assertion here.
    """

    def slice_features(self, imgs: Tensor, read_size: int | None = None) -> Tensor:
        """One `WIDTH`-wide vector per slice, carrying the slice's mean so it is traceable."""
        batch, slots, group = imgs.shape[:3]
        mean = imgs.float().mean(dim=(3, 4)).reshape(batch, slots, group, 1)
        return mean.expand(batch, slots, group, WIDTH) + float(read_size or 0) / 1000.0

    @override
    def forward(self, imgs: Tensor) -> Tensor:
        """Unused; the cache calls `slice_features` directly, as the trainer does."""
        return self.slice_features(imgs)


def _corpus() -> CachedCorpus:
    rng = np.random.default_rng(2026)
    images = rng.integers(0, 255, (STUDIES, SLOTS, SLICES, PIXELS, PIXELS), dtype=np.uint8)
    mask = np.ones((STUDIES, SLOTS), dtype=np.float32)
    mask[-1, -1] = 0.0
    return CachedCorpus(images, mask)


def _identifiers() -> list[str]:
    return [f"study{index}" for index in range(STUDIES)]


def test_key():
    """The name states everything that decides the contents, INCLUDING which encoder produced them.

    Features carry no visible sign of their encoder: a DINOv3 file and a DINOv2 file of the same shape
    load into the same head and score differently for no discoverable reason. The read size is separate
    from the cache's own size because our best recipe holds 448 px and reads it at 392.
    """
    assert EmbeddingCache(Path("."), SPEC).key == "small_336px_2sl_r336_cls_mean_max"

    # THE SOURCE CACHE'S OWN NAME, so every axis that decides the pixels reaches the features. This
    # restated three of that cache's six axes, which left the crop, the band and the tie rule with
    # nowhere to land: features encoded from the 80 mm arrays would have overwritten features encoded
    # from the 130 mm ones, at the right shape, under the same name.
    tighter = EmbeddingCache(Path("."), EmbeddingSpec("v3small", "448px_2sl_80mm_tissue", 384, "cls_mean", "max"))
    assert tighter.key == "v3small_448px_2sl_80mm_tissue_r384_cls_mean_max"

    # HOW THE SLICES REACHED THE ENCODER, because a whole-stack reading consumes them inside it and
    # stores one vector per slot: an array of different RANK from the same pixels, under what used to
    # be the same name.
    volume = EmbeddingCache(Path("."), EmbeddingSpec("small", f"336px_{SLICES}sl", 336, "cls_mean", "channels"))
    assert volume.key != EmbeddingCache(Path("."), SPEC).key
    assert EmbeddingCache(Path("."), SPEC).key != tighter.key


def test_features_path(tmp_path: Path):
    """The features live under the key, so two configurations cannot overwrite each other."""
    assert EmbeddingCache(tmp_path, SPEC).features_path == tmp_path / "features_small_336px_2sl_r336_cls_mean_max.npy"


def test_mask_path(tmp_path: Path):
    """The mask is written beside the features so the pair cannot be separated."""
    cache = EmbeddingCache(tmp_path, SPEC)
    assert cache.mask_path.parent == cache.features_path.parent
    assert cache.mask_path != cache.features_path


def test_index_path(tmp_path: Path):
    """The study order is an artefact too — without it a feature row cannot be aligned to a label."""
    assert EmbeddingCache(tmp_path, SPEC).index_path.suffix == ".json"


def test_exists(tmp_path: Path):
    """All three artefacts, not just the features: a matrix nobody can align to labels is not a cache."""
    cache = EmbeddingCache(tmp_path, SPEC)
    assert not cache.exists()
    cache.build(_FakeEncoder().slice_features, _corpus(), _identifiers(), torch.device("cpu"), batch=2)
    assert cache.exists()

    cache.index_path.unlink()
    assert not cache.exists(), "features without their index must not read as a usable cache"


def test_build(tmp_path: Path):
    """What is written is what the encoder produced, for the right study, in the right place.

    Checked against a recomputation rather than against a shape: a cache that transposed its axes, or
    wrote every batch to the same rows, has the right shape and is worthless.
    """
    cache = EmbeddingCache(tmp_path, SPEC)
    corpus = _corpus()
    written = cache.build(_FakeEncoder().slice_features, corpus, _identifiers(), torch.device("cpu"), batch=2)

    assert written.features.shape == (STUDIES, SLOTS, SLICES, WIDTH)
    expected = _FakeEncoder().slice_features(torch.from_numpy(corpus.images), SPEC.read_size)
    # float16 STORAGE, so the tolerance is the format's own resolution at this magnitude (~0.06 near
    # 130), not a fudge factor. These feed an autocast forward that would cast them down anyway.
    assert written.features.dtype == np.float16
    assert np.allclose(np.asarray(written.features, dtype=np.float32), expected.numpy(), rtol=1e-3, atol=0.1)
    assert np.array_equal(written.mask, corpus.mask), "the mask travels with the features"


def test_load(tmp_path: Path):
    """A built cache reads back identically, and an absent one raises rather than returning empty."""
    cache = EmbeddingCache(tmp_path, SPEC)
    with pytest.raises(FileNotFoundError, match="build it first"):
        cache.load()

    built = cache.build(_FakeEncoder().slice_features, _corpus(), _identifiers(), torch.device("cpu"), batch=3)
    loaded, identifiers = cache.load()
    assert identifiers == _identifiers()
    assert np.array_equal(np.asarray(loaded.features), np.asarray(built.features))
    assert np.array_equal(loaded.mask, built.mask)


def test_width(tmp_path: Path):
    """The width is read off the array, because the model is the authority on it, not a registry."""
    corpus = _corpus()
    cache = EmbeddingCache(tmp_path, SPEC)
    built = cache.build(_FakeEncoder().slice_features, corpus, _identifiers(), torch.device("cpu"))
    assert built.width == WIDTH


def test_slices(tmp_path: Path):
    """How many slices were encoded bounds what a pooling can later read over."""
    cache = EmbeddingCache(tmp_path, SPEC)
    built = cache.build(_FakeEncoder().slice_features, _corpus(), _identifiers(), torch.device("cpu"))
    assert built.slices == SLICES


def test_embedded_corpus_carries_its_mask():
    """A feature vector for a slot that was never acquired is zeros, and only the mask can say so."""
    features = np.zeros((STUDIES, SLOTS, SLICES, WIDTH), dtype=np.float16)
    mask = np.zeros((STUDIES, SLOTS), dtype=np.float32)
    corpus = EmbeddedCorpus(features, mask)
    assert corpus.width == WIDTH
    assert corpus.slices == SLICES
