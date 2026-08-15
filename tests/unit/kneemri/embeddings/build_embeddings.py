"""The encode entrypoint: what it refuses, and that it refuses before it costs anything."""

import pytest

from kneemri.embeddings.build_embeddings import main


def test_main_refuses_a_size_the_patch_grid_cannot_tile(monkeypatch: pytest.MonkeyPatch):
    """392 tiles at patch 14 and not at 16, and the refusal must come BEFORE any work.

    Checked before the data root is even read, because the alternative is discovering it after a
    multi-GB pixel cache is memory-mapped and an encoder is on the card — or not discovering it, since
    a transformers backbone may interpolate its position grid and hand back a plausible answer computed
    on a geometry the weights never saw.
    """
    monkeypatch.setattr("sys.argv", ["build_embeddings", "--variant", "v3small", "--image-size", "392"])
    with pytest.raises(ValueError, match="use 384 or 400"):
        main()


def test_main_accepts_a_size_that_tiles(monkeypatch: pytest.MonkeyPatch):
    """The same size is legal for DINOv2, so the guard is the ENCODER's and not a global rule.

    It fails later here — there is no pixel cache in a unit test — and that is the point: the tiling
    check is passed and the next thing that stops it is the absence of data, not the geometry.
    """
    monkeypatch.setattr("sys.argv", ["build_embeddings", "--variant", "small", "--image-size", "392"])
    with pytest.raises((FileNotFoundError, OSError)):
        main()
