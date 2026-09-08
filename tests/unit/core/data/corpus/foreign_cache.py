"""His decode served in our shape, and every way the join between the two can go wrong."""

from pathlib import Path

import numpy as np
import pytest

from core.data.corpus.foreign_cache import IMAGE_SIZE, SLICES, ForeignCache, ForeignShard

STUDIES = ("s0", "s1", "s2")
EXTRA = ("s3",)


def _shard(directory: Path, name: str, identifiers: tuple[str, ...], fill: int) -> None:
    """One half of his corpus written at his real shape, so the loader's own constants are exercised."""
    count = len(identifiers)
    np.save(directory / f"{name}_ids.npy", np.array(identifiers))
    volumes = np.full((count, SLICES, IMAGE_SIZE, IMAGE_SIZE), fill, dtype=np.uint8)
    for row in range(count):
        volumes[row, 0, 0, 0] = row
    np.save(directory / f"{name}_vols.npy", volumes)
    np.save(directory / f"{name}_masks.npy", np.ones((count, SLICES), dtype=np.uint8))


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Both shards, disjoint by identifier."""
    _shard(tmp_path, "all", STUDIES, fill=7)
    _shard(tmp_path, "extra", EXTRA, fill=9)
    return tmp_path


def test_ids_path(tmp_path: Path) -> None:
    """The identifier file is named for the shard."""
    assert ForeignShard(tmp_path, "all").ids_path() == tmp_path / "all_ids.npy"


def test_vols_path(tmp_path: Path) -> None:
    """The pixels are named for the shard."""
    assert ForeignShard(tmp_path, "extra").vols_path() == tmp_path / "extra_vols.npy"


def test_masks_path(tmp_path: Path) -> None:
    """The presence mask is named for the shard."""
    assert ForeignShard(tmp_path, "all").masks_path() == tmp_path / "all_masks.npy"


def test_foreign_shard_identifiers(corpus: Path) -> None:
    """A shard reads its own studies in storage order."""
    assert ForeignShard(corpus, "all").identifiers() == list(STUDIES)


def test_foreign_cache_identifiers(corpus: Path) -> None:
    """The cache reads both shards, concatenated in shard order."""
    assert ForeignCache(corpus).identifiers() == [*STUDIES, *EXTRA]


def test_volumes(corpus: Path) -> None:
    """The pixels come back memory-mapped at his shape, not read whole."""
    volumes = ForeignShard(corpus, "all").volumes()
    assert volumes.shape == (len(STUDIES), SLICES, IMAGE_SIZE, IMAGE_SIZE)
    assert volumes.dtype == np.uint8


def test_masks(corpus: Path) -> None:
    """The mask is float whatever width it was written at — here it was written as uint8."""
    masks = ForeignShard(corpus, "all").masks()
    assert masks.dtype == np.float32
    assert masks.shape == (len(STUDIES), SLICES)


def test_shards(corpus: Path) -> None:
    """Both halves, in the order their identifiers concatenate."""
    assert [shard.name for shard in ForeignCache(corpus).shards()] == ["all", "extra"]


def test_positions(corpus: Path) -> None:
    """Each wanted study is placed in its own shard and row, and an absent one is REFUSED, not dropped."""
    placed = ForeignCache(corpus).positions(["s3", "s1"])
    assert placed["s3"][0].name == "extra"
    assert placed["s3"][1] == 0
    assert placed["s1"][0].name == "all"
    assert placed["s1"][1] == 1

    with pytest.raises(KeyError, match="1 of 2 identifiers are absent"):
        ForeignCache(corpus).positions(["s1", "nobody"])


def test_load(corpus: Path) -> None:
    """His pixels for exactly these studies, in this order, with a slot axis our trainer can read."""
    images, order = ForeignCache(corpus).load(["s3", "s0"])

    assert order == ["s3", "s0"]
    assert images.images.shape == (2, 1, SLICES, IMAGE_SIZE, IMAGE_SIZE)
    assert images.images.dtype == np.uint8
    assert images.mask.shape == (2, 1)
    # s3 is the extra shard's row 0 and s0 the all shard's row 0: the fill separates the shards and the
    # corner separates the rows, so a load that crossed either would not produce these two.
    assert images.images[0, 0, 1, 0, 0] == 9
    assert images.images[1, 0, 1, 0, 0] == 7
    assert images.mask.tolist() == [[1.0], [1.0]]
