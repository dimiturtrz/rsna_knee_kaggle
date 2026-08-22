from pathlib import Path

import numpy as np

from core.data.corpus.window_reader import FileReader, MappedReader


def test_file_reader_read(tmp_path: Path):
    """The explicit read returns what the memory map returns, for every window a member can ask for.

    This is the whole contract: the file reader exists to change the SPEED of the fetch and nothing
    else, so it is judged against the indexing it replaces rather than against a hand-written expectation.
    Both the whole-stack path (one seek per study) and the windowed one (one seek per slot) are covered,
    and an unsorted batch, because `Trainer.batches` shuffles and does not sort within a batch.
    """
    generator = np.random.default_rng(0)
    written = generator.integers(0, 255, size=(9, 3, 6, 5, 5), dtype=np.uint8)
    path = tmp_path / "images.npy"
    np.save(path, written)
    mapped = np.load(path, mmap_mode="r")

    explicit, indexed = FileReader(mapped, path, threads=2), MappedReader(mapped)
    for batch, start, width in [
        (np.array([0, 1, 2]), 0, 6),
        (np.array([7, 2, 5, 0]), 2, 3),
        (np.array([8]), 5, 1),
        (np.array([3, 3]), 1, 4),
    ]:
        window = explicit.read(batch, start, width)
        assert window.shape == (len(batch), 3, width, 5, 5)
        assert np.array_equal(window, indexed.read(batch, start, width))
        assert np.array_equal(window, written[batch, :, start : start + width])


def test_mapped_reader_read(tmp_path: Path):
    """Indexing reads the same window, and reads only the slices asked for rather than all of them."""
    written = np.random.default_rng(2).integers(0, 255, size=(5, 2, 6, 4, 4), dtype=np.uint8)
    np.save(tmp_path / "images.npy", written)
    batch = np.array([4, 1])

    window = MappedReader(np.load(tmp_path / "images.npy", mmap_mode="r")).read(batch, 2, 3)
    assert window.shape == (2, 2, 3, 4, 4)
    assert np.array_equal(window, written[batch, :, 2:5])


def test_per_slot(tmp_path: Path):
    """One slot's slice, which is the unit a narrow window seeks by."""
    np.save(tmp_path / "images.npy", np.zeros((5, 2, 6, 4, 3), dtype=np.uint8))
    assert FileReader(np.load(tmp_path / "images.npy", mmap_mode="r"), tmp_path / "images.npy").per_slot == 12


def test_per_study(tmp_path: Path):
    """One study, which is what its position in the cache is a multiple of."""
    np.save(tmp_path / "images.npy", np.zeros((5, 2, 6, 4, 3), dtype=np.uint8))
    assert FileReader(np.load(tmp_path / "images.npy", mmap_mode="r"), tmp_path / "images.npy").per_study == 2 * 6 * 12
