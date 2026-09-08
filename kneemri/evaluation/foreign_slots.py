"""Our own pixels through the shifted path: `kneemri.evaluation.foreign_slots`.

The control half of `kneemri.evaluation.foreign_readout`. It lives beside `foreign_blocks` rather than
inside it because the two slabs answer different questions — this one prices the PLUMBING and that one
prices the PIXELS — and a file that holds both invites a reader to compare them as variants of one thing.
"""

from typing import override

import numpy as np
from numpy.typing import NDArray

from core.data.corpus.study_cache import CachedCorpus
from kneemri.inference.holdout_slab import HoldoutSlab


class SlotsKept(HoldoutSlab):
    """Our own pixels carrying only the slots his decode has a correspondence for.

    Under `foreign_blocks.BLOCKS` that is all six, so this is a plumbing self-check whose score must equal
    the cached native one. It stays a real slab rather than an identity because the correspondence is a
    measurement and can narrow: a future block map that drops a slot has its cost priced here rather than
    silently folded into the pixel number.

    Blanking happens per batch rather than over a copied array: the holdout is 12.8 GB at 336 px and a
    batch of eight is 60 MB. A withheld slot is zeros with its mask entry at 0.0, which is exactly how
    `study.py` represents an acquisition a study never routed — a shape the head's softmax already
    handles rather than an input it has never seen.
    """

    def __init__(self, corpus: CachedCorpus, holdout: NDArray[np.int64], kept: tuple[int, ...]) -> None:
        super().__init__(corpus, holdout)
        self.kept = kept

    @override
    def batch(self, begin: int, size: int) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        """The same studies this slab would serve, carrying only the kept slots."""
        images, mask = super().batch(begin, size)
        kept = np.zeros_like(images)
        presence = np.zeros_like(mask)
        for slot in self.kept:
            kept[:, slot] = images[:, slot]
            presence[:, slot] = mask[:, slot]
        return kept, presence
