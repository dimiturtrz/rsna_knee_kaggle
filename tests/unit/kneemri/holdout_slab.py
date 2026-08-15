"""The holdout slab, whose whole contract is that it reads nothing until asked and then only a batch."""

import numpy as np

from kneemri.holdout_slab import HoldoutSlab
from kneemri.study_cache import CachedCorpus

STUDIES, SLOTS = 4, 3


def _corpus() -> CachedCorpus:
    images = np.arange(STUDIES * SLOTS).reshape(STUDIES, SLOTS, 1, 1, 1).astype(np.uint8)
    return CachedCorpus(images, np.arange(STUDIES * SLOTS).reshape(STUDIES, SLOTS).astype(np.float32))


def test_count():
    """Answered from the holdout's own length, so a consumer can size its walk without a read."""
    assert HoldoutSlab(_corpus(), np.array([1, 3])).count == 2


def test_batch():
    """A batch is the holdout's rows, in the holdout's order — pixels and mask taken over the same ones.

    Order is the part worth asserting: the rows come back in the order the caller asked for, not the
    corpus's, because the predictions are written against a study list in that same order.
    """
    slab = HoldoutSlab(_corpus(), np.array([3, 1]))

    images, mask = slab.batch(0, 2)

    assert images.reshape(2, SLOTS).tolist() == [[9, 10, 11], [3, 4, 5]]
    assert mask.reshape(2, SLOTS).tolist() == [[9.0, 10.0, 11.0], [3.0, 4.0, 5.0]]


def test_batches():
    """Every study exactly once, in batches of the asked size with a short one at the end.

    A dropped tail is the failure that matters here: it produces a prediction array shorter than the
    study list it is written beside, and both are plausible shapes.
    """
    slab = HoldoutSlab(_corpus(), np.array([0, 1, 2, 3]))

    shapes = [images.shape[0] for images, _ in slab.batches(3)]
    seen = np.concatenate([images.reshape(-1, SLOTS)[:, 0] for images, _ in slab.batches(3)])

    assert shapes == [3, 1]
    assert seen.tolist() == [0, 3, 6, 9], "every study, once, in holdout order"


def test_of():
    """Pixels a caller already holds — occlusion builds its own — reach the same batching path."""
    images = np.arange(2 * SLOTS).reshape(2, SLOTS, 1, 1, 1).astype(np.uint8)

    slab = HoldoutSlab.of(images, np.ones((2, SLOTS), np.float32))

    assert slab.count == 2
    assert slab.batch(1, 1)[0].reshape(SLOTS).tolist() == [3, 4, 5]
