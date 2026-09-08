"""The foreign pixel path served as one of our caches: `core.data.corpus.foreign_cache`.

WHY THIS EXISTS. Every offline number this project owns is read on 909 studies drawn from OUR corpus,
through OUR decode, under OUR label table — the same three things every candidate was fitted under. So
an offline seat ADD asks only "do you add on studies you were built to fit?", which is an
IN-DISTRIBUTION redundancy measure, and sixteen candidates spanning every axis we own have now priced
negative on it while both seats that ever moved the board were out-of-distribution members. Nothing here
has ever measured generalisation.

`raptor-corpus` is the instrument that can. It covers ALL 4,407 of our studies and no others (3,200 base
+ 1,207 extra, verified by identifier), at (44, 336, 336) uint8 — **a different pixel path over the same
studies with the same labels**. Scoring a member we already own on these pixels holds the corpus and the
supervision fixed and moves the decode alone, which is a distribution shift with a control, and it costs
no training.

WHAT A NUMBER READ HERE PRICES, STATED SO THE NULL CANNOT BE MISREAD. His 44 slices already encode HIS
span and HIS windowing. A member's drop between our pixels and these therefore prices **his pixel path as
a whole** against ours — never resampling alone, and never interpolation. A flat reading means the two
paths agree on what this member looks at, not that slice geometry is irrelevant.

WHY NO RESAMPLING STEP. Our trainer reads a window of `width` slices from `start` inside a slot, so 44
slices in one slot is already the shape it consumes; a resample to our own slice count would ADD a
transform that is ours to the pixels that are his, confounding the one variable this cache exists to
move. The band is expressed as the window, exactly as it is on our own cache.

THE SUBSET IS THE POINT. The full corpus is ~22 GB across the two files. A reading is taken on the 909
holdout studies, so this materialises only the identifiers asked for (~4.5 GB) rather than mapping the
whole thing — and it refuses rather than silently serving a short frame when an identifier is absent,
because a member scored on a quietly narrower population is the denominator defect this repo has already
paid for three times.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float, UInt8
from numpy.typing import NDArray

from core.data.corpus.study_cache import CachedCorpus

logger = logging.getLogger(__name__)

SHARDS = ("all", "extra")
"""His two halves. They are disjoint by identifier and together are exactly our corpus."""

SLICES = 44
IMAGE_SIZE = 336
SLOTS = 1
"""The STORAGE shape only: his file is `(studies, 44, 336, 336)` and the slot dimension is inserted as 1
so the return type matches `StudyCache.load`.

His 44 is NOT one acquisition. D206 measured `all_masks.npy` filling at 1.000 / 0.973 / 1.000 / 0.742 /
1.000 on columns 0-11 / 12-21 / 22-29 / 30-35 / 36-43 — five blocks, two of them partial, which is a slot
scheme and not a flat stack; adjacent-slice correlation runs 0.8-0.9 inside a block and -0.22 across the
35->36 boundary. The blocks map onto our six slots as a fixed permutation, and splicing them is
`kneemri.evaluation.foreign_readout.BLOCKS`, not this loader's business: a cache serves what is on disk."""


@dataclass(frozen=True)
class ForeignShard:
    """One half of his corpus: the identifiers, the volumes, and the presence mask beside them."""

    directory: Path
    name: str

    def ids_path(self) -> Path:
        """Where this shard's identifiers live."""
        return self.directory / f"{self.name}_ids.npy"

    def vols_path(self) -> Path:
        """Where this shard's pixels live."""
        return self.directory / f"{self.name}_vols.npy"

    def masks_path(self) -> Path:
        """Where this shard's presence mask lives."""
        return self.directory / f"{self.name}_masks.npy"

    def identifiers(self) -> list[str]:
        """Every study this shard holds, in the order its volumes are stored."""
        return [str(identifier) for identifier in np.load(self.ids_path())]

    def volumes(self) -> UInt8[np.ndarray, "studies slices height width"]:
        """The pixels, memory-mapped — the file is gigabytes and a reading wants a few hundred studies."""
        return np.load(self.vols_path(), mmap_mode="r")

    def masks(self) -> Float[np.ndarray, "studies slices"]:
        """The presence mask as a float, whatever integer or boolean width it was written at."""
        return np.asarray(np.load(self.masks_path()), dtype=np.float32)


@dataclass(frozen=True)
class ForeignCache:
    """His decode of our corpus, served in the shape our trainer already consumes.

    `load` returns the same `(CachedCorpus, identifiers)` pair `StudyCache.load` does, so a consumer
    reads one or the other without knowing which pixel path it got — which is what makes the drop
    between them a measurement rather than a comparison of two programs.
    """

    directory: Path

    def shards(self) -> tuple[ForeignShard, ...]:
        """Both halves, in the order their identifiers are concatenated."""
        return tuple(ForeignShard(self.directory, name) for name in SHARDS)

    def identifiers(self) -> list[str]:
        """Every study his corpus holds, both shards, in shard order."""
        return [identifier for shard in self.shards() for identifier in shard.identifiers()]

    def positions(self, wanted: list[str]) -> dict[str, tuple[ForeignShard, int]]:
        """Where each wanted identifier sits: which shard, and its row inside that shard.

        Refuses on an identifier his corpus does not hold rather than dropping it, because a member
        scored on a quietly narrower frame reports a number about a population nobody named.
        """
        found: dict[str, tuple[ForeignShard, int]] = {}
        for shard in self.shards():
            for row, identifier in enumerate(shard.identifiers()):
                if identifier in found:
                    message = f"{identifier} appears in more than one shard - the two are not disjoint"
                    raise ValueError(message)
                found[identifier] = (shard, row)
        missing = [identifier for identifier in wanted if identifier not in found]
        if missing:
            message = (
                f"{len(missing)} of {len(wanted)} identifiers are absent from {self.directory} ({missing[0]}, ...)"
            )
            raise KeyError(message)
        return {identifier: found[identifier] for identifier in wanted}

    def load(self, wanted: list[str]) -> tuple[CachedCorpus, list[str]]:
        """His pixels for exactly these studies, in this order, shaped as our trainer reads them."""
        placed = self.positions(wanted)
        images: NDArray[np.uint8] = np.empty((len(wanted), SLOTS, SLICES, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        mask: NDArray[np.float32] = np.empty((len(wanted), SLOTS), dtype=np.float32)
        opened = {shard.name: (shard.volumes(), shard.masks()) for shard in self.shards()}
        for index, identifier in enumerate(wanted):
            shard, row = placed[identifier]
            volumes, presence = opened[shard.name]
            images[index, 0] = volumes[row]
            mask[index, 0] = float(np.asarray(presence[row]).mean())
        logger.info("served %d studies from %s at %d slices, %d px", len(wanted), self.directory, SLICES, IMAGE_SIZE)
        return CachedCorpus(images=images, mask=mask), list(wanted)
