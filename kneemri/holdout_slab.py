"""The holdout's pixels, taken out of the memory-mapped study cache only when something needs them."""

import numpy as np
from numpy.typing import NDArray

from kneemri.study_cache import CachedCorpus


class HoldoutSlab:
    """The holdout's pixels, copied out of the memory-mapped cache only if something actually asks.

    The copy is several gigabytes — 881 studies at six slots, twelve slices and 336 pixels — so a run
    answered entirely from cached probabilities should never make it. Lazy rather than skipped, because
    one member of a group missing is enough to need it, and once made it serves every member of that
    group rather than being paid per member.
    """

    def __init__(self, corpus: CachedCorpus, holdout: NDArray[np.int64]) -> None:
        self.corpus = corpus
        self.holdout = holdout
        self.held: tuple[NDArray[np.uint8], NDArray[np.float32]] | None = None

    def arrays(self) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        """The holdout's images and presence mask, materialised on the first call and kept."""
        held = self.held or (self.corpus.images[self.holdout], self.corpus.mask[self.holdout])
        self.held = held
        return held
