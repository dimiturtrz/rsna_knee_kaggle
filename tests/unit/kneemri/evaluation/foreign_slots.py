"""The control slab: our own pixels restricted to the slots his decode corresponds to."""

import numpy as np

from core.data.corpus.study_cache import CachedCorpus
from kneemri.evaluation.foreign_slots import SlotsKept

SLOTS = 3
SLICES = 4
SIZE = 2


def _corpus(slots: int, fill: int) -> CachedCorpus:
    """A tiny corpus whose slot index is written into its own pixels, so a misplacement is visible."""
    images = np.zeros((2, slots, SLICES, SIZE, SIZE), dtype=np.uint8)
    for slot in range(slots):
        images[:, slot] = fill + slot
    return CachedCorpus(images=images, mask=np.ones((2, slots), dtype=np.float32))


def test_batch() -> None:
    """A kept slot is untouched and an unkept one is BLANKED, not merely unmasked."""
    holdout = np.arange(2, dtype=np.int64)

    kept, presence = SlotsKept(_corpus(SLOTS, fill=10), holdout, kept=(0, 2)).batch(0, 2)

    assert kept.shape == (2, SLOTS, SLICES, SIZE, SIZE)
    assert kept[0, 0, 0, 0, 0] == 10, "a kept slot is untouched"
    assert kept[0, 2, 0, 0, 0] == 12
    assert kept[:, 1].max() == 0, "an unkept slot is blanked, not merely unmasked"
    assert presence.tolist() == [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
