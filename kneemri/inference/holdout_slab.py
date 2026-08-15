"""The holdout's pixels, handed to a model one batch at a time instead of all at once."""

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from kneemri.corpus.study_cache import CachedCorpus


class HoldoutSlab:
    """The holdout's pixels as a SOURCE of batches, never as one array.

    It used to copy `corpus.images[holdout]` whole and hand that over: 881 studies at six slots, twelve
    slices and 336 px is 12.8 GB, and 22.8 GB at 448 px. Every consumer then walked the copy in batches
    of eight, so the whole array was resident for the duration of a walk that never needed more than one
    batch of it — and the gather itself is the slow way to read those bytes, since a memory map serves
    them as page faults where `CachedCorpus.window` reads them explicitly at roughly twice the rate.

    Batching here rather than in each consumer keeps the read policy in ONE place: the corpus already
    decides between an explicit file read and plain indexing, and a slab that materialised first threw
    that decision away.

    Nothing is read at construction, which is what lets an analysis answered entirely from cached
    probabilities touch no pixels at all.
    """

    def __init__(self, corpus: CachedCorpus, holdout: NDArray[np.int64]) -> None:
        self.corpus = corpus
        self.holdout = holdout

    @classmethod
    def of(cls, images: NDArray[np.uint8], mask: NDArray[np.float32]) -> "HoldoutSlab":
        """A slab over arrays already in hand, for a caller that holds pixels rather than a corpus.

        Occlusion diagnostics are the case: they build their own images by blanking regions, so there
        is no cache to read them from. `MappedReader` serves those by plain indexing, so this costs
        the batching path nothing and saves a second spelling of "the pixels to predict over".
        """
        return cls(CachedCorpus(images, mask), np.arange(len(images), dtype=np.int64))

    @property
    def count(self) -> int:
        """How many studies the slab covers, without reading any of them."""
        return len(self.holdout)

    def batch(self, begin: int, size: int) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        """`size` studies from `begin`: every slot and every slice, read through the corpus.

        The full slice extent rather than a window, because a member averages over several window
        starts and each start reads the same studies — so the batch is read once and sliced in memory.
        A whole-stack read is also the file reader's fastest shape: one seek per study.
        """
        positions = self.holdout[begin : begin + size]
        return self.corpus.window(positions, 0, self.corpus.slices), self.corpus.mask[positions]

    def batches(self, size: int) -> Iterator[tuple[NDArray[np.uint8], NDArray[np.float32]]]:
        """Every batch in holdout order, so a consumer's rows come back in the order it asked for."""
        for begin in range(0, self.count, size):
            yield self.batch(begin, size)
