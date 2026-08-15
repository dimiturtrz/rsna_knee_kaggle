from pathlib import Path

import numpy as np
import torch

from kneemri.corpus.prefetch import Prefetch
from kneemri.corpus.study_cache import CachedCorpus


def corpus_of(tmp_path: Path, studies: int = 9) -> CachedCorpus:
    """A file-backed corpus, so the reader under test is the one training actually uses."""
    written = np.random.default_rng(4).integers(0, 255, size=(studies, 2, 6, 4, 4), dtype=np.uint8)
    np.save(tmp_path / "images.npy", written)
    return CachedCorpus(np.load(tmp_path / "images.npy", mmap_mode="r"), np.ones((studies, 2), dtype=np.float32))


def test_fetch(tmp_path: Path):
    """One scheduled step's pixels are the window its plan entry names, and nothing else."""
    corpus = corpus_of(tmp_path)
    plan = [(np.array([1, 4]), 2), (np.array([0, 8]), 0)]

    reader = Prefetch(corpus, plan, 3, pin=False)
    assert torch.equal(reader.fetch(1), torch.from_numpy(corpus.window(np.array([0, 8]), 0, 3)))
    assert reader.fetch(0).shape == (2, 2, 3, 4, 4)


def test_iter(tmp_path: Path):
    """Reading a step ahead yields exactly the serial stream — same order, same bytes, same length.

    This is the whole safety claim: a prefetched epoch must be indistinguishable from the one the loop
    would have read for itself, or the schedule is quietly training on something else.
    """
    corpus = corpus_of(tmp_path)
    generator = np.random.default_rng(6)
    plan = [(generator.permutation(9)[:3], int(generator.integers(0, 4))) for _ in range(5)]

    served = list(Prefetch(corpus, plan, 3, pin=False))
    assert len(served) == len(plan)
    for (batch, pixels), (expected, start) in zip(served, plan, strict=True):
        assert np.array_equal(batch, expected)
        assert torch.equal(pixels, torch.from_numpy(corpus.window(expected, start, 3)))


def test_iter_serves_an_empty_plan(tmp_path: Path):
    """No steps means no reads — an epoch with nothing scheduled must not block on a fetch."""
    assert list(Prefetch(corpus_of(tmp_path), [], 3, pin=False)) == []
