"""The volumetric encode entrypoint: the name its cache lands under, and that it stops on missing data."""

from pathlib import Path

import pytest

from kneemri.embeddings.build_ortho_embeddings import VARIANT, VOLUME, main
from kneemri.embeddings.embedding_cache import EmbeddingCache, EmbeddingSpec
from kneemri.models.ortho_diffusion import MIDDLE_WIDTH, READ_SIZE

SOURCE = "images_420px_12sl_160mm"
"""A pixel-cache key of the shape this build reads, so the composed name is the real one."""


def test_main_stops_on_a_missing_pixel_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """No weights are touched before the corpus is: a unit test may not reach either, and must not."""
    monkeypatch.setattr("sys.argv", ["build_ortho_embeddings", "--image-size", "420", "--slices", "12"])
    with pytest.raises((FileNotFoundError, OSError)):
        main()


def test_the_written_key_names_the_whole_stack_reading() -> None:
    """The cache filename has to say VOLUME, because its rank differs from every per-slice cache."""
    spec = EmbeddingSpec(
        variant=VARIANT, source=SOURCE, read_size=READ_SIZE, pool=f"gap{MIDDLE_WIDTH}", slice_read=VOLUME
    )
    key = EmbeddingCache(Path(), spec).key
    assert key.startswith(VARIANT)
    assert key.endswith(VOLUME)
