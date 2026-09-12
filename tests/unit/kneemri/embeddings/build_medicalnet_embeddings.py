"""The second volumetric entrypoint: that its key records the 2048-wide read, and that it needs data."""

from pathlib import Path

import pytest

from kneemri.embeddings.build_medicalnet_embeddings import VARIANT, VOLUME, main
from kneemri.embeddings.embedding_cache import EmbeddingCache, EmbeddingSpec
from kneemri.models.medicalnet import READ_SIZE, WIDTH

SOURCE = "images_336px_12sl_160mm"
"""The pixel-cache key this build was run against, so the composed name is the one on disk."""


def test_main_stops_on_a_missing_pixel_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """The corpus is opened before the checkpoint is, which is what keeps this test weight-free."""
    monkeypatch.setattr("sys.argv", ["build_medicalnet_embeddings", "--image-size", "97", "--slices", "3"])
    with pytest.raises((FileNotFoundError, OSError)):
        main()


def test_the_key_separates_this_trunk_from_the_other_volumetric_cache() -> None:
    """Same source, same size, same reading — so the variant and the 2048 are the whole difference."""
    spec = EmbeddingSpec(variant=VARIANT, source=SOURCE, read_size=READ_SIZE, pool=f"gap{WIDTH}", slice_read=VOLUME)
    key = EmbeddingCache(Path(), spec).key
    assert key.startswith(VARIANT)
    assert f"gap{WIDTH}" in key
    assert key.endswith(VOLUME)
