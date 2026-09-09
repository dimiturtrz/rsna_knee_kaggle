"""The block splice: his flat stack resampled and laid into our slot layout."""

import numpy as np

from core.data.corpus.study_cache import CachedCorpus
from kneemri.evaluation.foreign_blocks import BLOCKS, KEPT, MappedSlab, SlotBlock

SLOTS = 3
SLICES = 4
SIZE = 2
HIS_SLICES = 8


def _ramp() -> CachedCorpus:
    """One his-style volume whose slice INDEX is its pixel value, so order and direction are readable."""
    images = np.zeros((2, 1, HIS_SLICES, SIZE, SIZE), dtype=np.uint8)
    for slice_ in range(HIS_SLICES):
        images[:, 0, slice_] = slice_
    return CachedCorpus(images=images, mask=np.ones((2, 1), dtype=np.float32))


def test_taken() -> None:
    """A block is resampled to our slice count, forwards — there is no reversal flag to exercise."""
    volumes = _ramp().images[:, 0]

    forward = SlotBlock(slot=0, begin=2, end=6).taken(volumes, SLICES)
    assert forward.shape == (2, SLICES, SIZE, SIZE)
    assert [int(forward[0, index, 0, 0]) for index in range(SLICES)] == [2, 3, 4, 5], "his own order, resampled"

    # Six of his slices onto four of ours is the real 30-35 case: the ends are pinned and nothing is
    # dropped off either edge, which is what makes the resample the free leg D205's control priced.
    narrowed = SlotBlock(slot=0, begin=0, end=6).taken(volumes, SLICES)
    assert int(narrowed[0, 0, 0, 0]) == 0
    assert int(narrowed[0, -1, 0, 0]) == 5


def test_blocks() -> None:
    """Every one of our six slots is served, and his five blocks tile his stack without overlap."""
    assert KEPT == (0, 1, 2, 3, 4, 5), "the splice leaves no slot absent"

    # Slot 5 deliberately repeats slot 3's block: our two are near-duplicates of one series (0.836
    # against a 0.115-0.396 background), so his single block has one of ours to match on either.
    spans = {(block.begin, block.end) for block in BLOCKS}
    assert len(spans) == len(BLOCKS) - 1
    assert sorted(spans) == [(0, 12), (12, 22), (22, 30), (30, 36), (36, 44)], "his five blocks, in his order"

    # His block order is NOT our slot order -- a positional splice would be wrong on three of five.
    assert [block.begin for block in BLOCKS] != sorted(block.begin for block in BLOCKS)

    # D209: the direction is a per-STUDY bit and it is ours, so no block carries a reversal constant.
    assert all(not hasattr(block, "reversed_") for block in BLOCKS)


def test_depth() -> None:
    """The slab declares OUR slice count, never his — the number a caller sizes its window against.

    This is the whole defect knee-kcvma was: the window was sized against `corpus.slices` (his 44) while
    `batch` serves `slices` (ours), so it indexed a stack nobody handed over. Asserting the two DIFFER
    is the point — an assertion that only checked the value would pass on a slab that had never been
    reshaped at all.
    """
    slab = MappedSlab(_ramp(), np.arange(2, dtype=np.int64), slots=SLOTS, slices=SLICES, blocks=BLOCKS[:1])

    assert slab.depth == SLICES
    assert slab.corpus.slices == HIS_SLICES
    assert slab.depth != slab.corpus.slices, "the served depth is not the source depth"
    assert slab.batch(0, 2)[0].shape[2] == slab.depth, "and what it serves is what it declares"


def test_batch() -> None:
    """His blocks land in the slot each corresponds to, and a slot no block maps to stays absent."""
    holdout = np.arange(2, dtype=np.int64)
    blocks = (SlotBlock(slot=2, begin=0, end=4), SlotBlock(slot=0, begin=4, end=8))

    placed, his = MappedSlab(_ramp(), holdout, slots=SLOTS, slices=SLICES, blocks=blocks).batch(0, 2)

    assert placed.shape == (2, SLOTS, SLICES, SIZE, SIZE), "his flat stack is expanded to our layout"
    assert [int(placed[0, 2, index, 0, 0]) for index in range(SLICES)] == [0, 1, 2, 3], "his first block, our slot 2"
    assert [int(placed[0, 0, index, 0, 0]) for index in range(SLICES)] == [4, 5, 6, 7], "his second, our slot 0"
    assert placed[:, 1].max() == 0, "a slot no block maps to stays absent"
    assert his.tolist() == [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
