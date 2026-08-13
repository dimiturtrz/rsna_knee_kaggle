"""The holdout slab, whose whole contract is that it copies nothing until asked and then copies once."""

import numpy as np

from kneemri.holdout_slab import HoldoutSlab
from kneemri.study_cache import CachedCorpus


def test_arrays():
    """The holdout's rows, copied once and kept — and not copied at all until asked.

    Laziness is the point rather than an optimisation: the copy is several GB of pixels, and a curve
    answered entirely from cached probabilities is exactly the run that must not pay it. Identity rather
    than equality on the second call, because a slab that recopied would still return equal arrays and
    the whole claim is that it does not recopy.
    """
    images = np.arange(4 * 3).reshape(4, 3, 1, 1, 1).astype(np.uint8)
    slab = HoldoutSlab(CachedCorpus(images, np.ones((4, 3), np.float32)), np.array([1, 3]))

    assert slab.held is None, "constructing the slab must not read anything"

    first = slab.arrays()

    assert (first[0].reshape(2, 3) == np.array([[3, 4, 5], [9, 10, 11]], dtype=np.uint8)).all()
    assert first[1].shape == (2, 3), "the mask is taken over the same rows as the pixels"
    assert slab.arrays()[0] is first[0], "the copy is made once, not once per member"
