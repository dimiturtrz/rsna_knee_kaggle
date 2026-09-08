"""His decode laid into our slot layout: `kneemri.evaluation.foreign_blocks`.

WHICH BLOCK IS WHICH SLOT. D206 §3 answered it as a per-study argmax and then as its distribution over
200 studies, which is the only form that can show constancy: four of his five blocks land on one of our
slots on 79-93% of studies, and the fifth splits between two of ours that are near-duplicates of each
other. D209 §5 refitted the same permutation on `holdout909` — the holdout itself, not a shard
intersection — and it holds with a smallest argmax margin of 0.599, so nothing here needs refitting.

WHY THERE IS NO REVERSAL FLAG, WRITTEN DOWN BECAUSE THERE USED TO BE ONE. `SlotBlock` carried a
`reversed_=True` constant on the first sagittal block, on the reading that his stack traverses the anatomy
the other way. Two measurements retired it:

  - The direction is a per-STUDY bit, not a per-block constant, and it is OURS and deliberate.
    `frac_reversed@sag_series = 0.0000` over 9864 of 9864 sagittal series says the archive stores every
    sagittal acquisition with the same through-plane normal; `Laterality.normalise` then flips the slice
    axis for a right knee, giving `frac_reversed@sag_series_cached = 0.4752` over 9651. That cached sign
    predicts the per-study correlation direction on 401 of 401 studies. So a constant is the wrong SHAPE,
    and the right one is free: `side == "R"` off `Study.side`.
  - Almost nothing can observe it. Of the six slice poolings in `kneemri.models.slice_pooling`, only
    `depth` and `channels` see slice ORDER at all; `max`, `attend`, `labelwise` and `moments` are
    permutation-invariant, and across the arms in this tree that is 8 of 366. A reversal the pool cannot
    see is machinery that can only be wrong.

So blocks are served forwards. An order-aware member that ever needs the sign takes it from `Study.side`
rather than from a flag fitted here.
"""

from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray

from core.data.corpus.study_cache import CachedCorpus
from kneemri.inference.holdout_slab import HoldoutSlab


@dataclass(frozen=True)
class SlotBlock:
    """Which of his slices stand in for one of our acquisition slots. `begin`/`end` are half-open."""

    slot: int
    begin: int
    end: int

    def taken(self, volumes: NDArray[np.uint8], slices: int) -> NDArray[np.uint8]:
        """His block for these studies, resampled to the slice count our slot carries.

        The resample is the one operation D205's positive control licenses outright: our own decode at
        twenty slices against our own at twelve reads 0.9999, so changing slice COUNT costs nothing on
        this instrument and cannot be what a drop measures.
        """
        block = volumes[:, self.begin : self.end]
        picked = np.linspace(0, block.shape[1] - 1, slices).round().astype(np.int64)
        return np.ascontiguousarray(block[:, picked])


BLOCKS = (
    SlotBlock(slot=0, begin=0, end=12),
    SlotBlock(slot=1, begin=22, end=30),
    SlotBlock(slot=2, begin=36, end=44),
    SlotBlock(slot=3, begin=12, end=22),
    SlotBlock(slot=4, begin=30, end=36),
    SlotBlock(slot=5, begin=12, end=22),
)
"""Modal share is 0.79-0.93 on four blocks, and his block order is NOT our slot order (his second block is
our fourth slot), so a positional splice would be wrong on three of the five.

His block 12-21 serves BOTH our slot 3 and our slot 5, and that is a fact about our pipeline rather than
about his: over studies where both are live those two slots correlate 0.836 against a 0.115-0.396
background for every other pair, because `slots.py` reads one series twice, wide and tight, so the head
pools over scale. They co-occur on 20 of 200 studies and are otherwise alternates, so his single block
genuinely has one of ours to match on almost every study."""

KEPT = tuple(block.slot for block in BLOCKS)


class MappedSlab(HoldoutSlab):
    """His decode laid into our slot layout block by block, the uncorresponded slots left absent.

    Expansion is per batch for the same reason `SlotsKept` blanks per batch: his corpus at six slots and
    our slice count would be tens of gigabytes materialised and is megabytes a batch.
    """

    def __init__(
        self, corpus: CachedCorpus, holdout: NDArray[np.int64], slots: int, slices: int, blocks: tuple[SlotBlock, ...]
    ) -> None:
        super().__init__(corpus, holdout)
        self.slots = slots
        self.slices = slices
        self.blocks = blocks

    @override
    def batch(self, begin: int, size: int) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        """His volume for these studies, block by block, in the slot each block corresponds to."""
        images, mask = super().batch(begin, size)
        studies, _, _, height, width = images.shape
        placed = np.zeros((studies, self.slots, self.slices, height, width), dtype=np.uint8)
        presence = np.zeros((studies, self.slots), dtype=np.float32)
        for block in self.blocks:
            placed[:, block.slot] = block.taken(images[:, 0], self.slices)
            presence[:, block.slot] = mask[:, 0]
        return placed, presence
