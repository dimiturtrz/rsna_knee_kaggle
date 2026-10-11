"""How a batch of studies reaches the trainer: by indexing the cache, or by reading its file outright.

The fetch is synchronous on the training thread, so its cost is wall-clock on every step, and on the
larger caches it is the step's biggest host-side term. Which reader a corpus uses is decided by whether
its pixels are file-backed at all — see `CachedCorpus.reader`.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

READERS = 4
"""Handles a batch is read through.

Two, until `fetch_bench` timed the branch the TRAINER actually reads. The original number came from a
whole-stack read — one seek per study — where two handles already saturate: 336 px cache, whole stack,
1.06 / 1.35 / 1.35 / 1.38 GB/s at 1 / 2 / 4 / 8. A member reads a NARROW WINDOW, which is one seek per
slot instead, and a seek-bound pattern keeps scaling past two: 0.72 / 1.06 / 1.39 / 1.31 GB/s over the
same handles, reproduced at 0.77 / 1.25 / 1.43 / 0.92 with the arms run in the opposite order over
different rows. Four is the peak of the read the trainer performs and costs nothing on the read it does
not, so it is the setting both branches want.

Worth 9% of a STEP, not only of the fetch: a 448 px epoch over 476 studies ran 22 / 23 / 21 s at two
handles against 20 / 20 / 20 s at four, the arms interleaved so page-cache warming could not name the
winner, with the loss identical to four decimals in all of them. Prefetch does not hide it. That 9% is
measured on studies small enough to be RESIDENT, so a full-corpus epoch — which reads colder — saves at
least this much (knee-lky3).
"""


@dataclass(frozen=True, kw_only=True)
class SliceSpan:
    """Which slices of every slot a read takes: `width` of them, from `start`, along the slice axis.

    `start` and `width` are both non-negative ints indexing the same axis, so a swapped pair is a valid
    read of the wrong slices that returns the right shape. Keyword-only construction is the guard; the
    type is what lets the pair travel as one argument instead of two that can be crossed at every hop.
    """

    start: int
    width: int

    @property
    def stop(self) -> int:
        """One past the last slice read — the end of the slice expression."""
        return self.start + self.width

    def covers(self, depth: int) -> bool:
        """Whether this span is the whole stack, so a study is one contiguous read rather than one per slot."""
        return self.width == depth


class WindowReader(Protocol):
    """The one thing a reader does: hand back a span of slices of every slot for a batch of positions."""

    def read(self, batch: NDArray[np.int64], span: SliceSpan) -> NDArray[np.uint8]: ...


class MappedReader:
    """Reads a batch's window by indexing the array itself — correct for anything, mapped or resident."""

    def __init__(self, images: NDArray[np.uint8]) -> None:
        self.images = images

    def read(self, batch: NDArray[np.int64], span: SliceSpan) -> NDArray[np.uint8]:
        """The window, contiguous, in ONE indexing op — never index the batch and slice afterwards."""
        return np.ascontiguousarray(self.images[batch, :, span.start : span.stop])


class FileReader:
    """Reads the same window straight out of the cache file, which a memory map does at half the speed.

    A memory map serves a 174 MB batch as some forty thousand 4 KB page faults, and the fault path — not
    the drive — is what the fetch waits on: measured on the 448 px cache, the gather ran at 0.88 GB/s
    where the same bytes read explicitly ran at 1.67, on two disjoint halves of the corpus tested both
    ways round so neither warmed the other. Splitting the batch over a few handles buys most of the rest
    (2.27 GB/s at two threads); past four it falls off again, so the pool is small on purpose.

    This changes no number the trainer produces. It returns the array the map would have returned, which
    is what the equality oracle in the tests asserts — the win is entirely in how the bytes arrive.
    """

    def __init__(self, images: np.memmap, path: Path, threads: int = READERS) -> None:
        self.path = path
        self.offset = images.offset
        self.shape = images.shape
        self.strides = images.strides
        self.dtype = images.dtype
        self.threads = threads
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="fetch")

    @property
    def per_slot(self) -> int:
        """Bytes in one slot's slice — the unit a window is built from."""
        return int(np.prod(self.shape[3:]))

    @property
    def per_study(self) -> int:
        """Bytes between one study and the next IN THE FILE — the map's stride, not its shape.

        A map served as the leading slots of a wider cache has fewer slots than the rows on disk hold,
        and a stride read off the shape would walk into the next study's pixels without a word.
        """
        return int(self.strides[0])

    def read(self, batch: NDArray[np.int64], span: SliceSpan) -> NDArray[np.uint8]:
        """The window for every position in the batch, the rows shared out over the handles."""
        window = np.empty((len(batch), self.shape[1], span.width, *self.shape[3:]), dtype=self.dtype)
        shares = [range(offset, len(batch), self.threads) for offset in range(self.threads)]
        list(self.pool.map(lambda rows: self._fill(window, batch, rows, span), shares))
        return window

    def _fill(
        self,
        window: NDArray[np.uint8],
        batch: NDArray[np.int64],
        rows: range,
        span: SliceSpan,
    ) -> None:
        """Fill this share of the rows through one handle of its own — handles are not thread-safe.

        A whole-stack read is one seek per study, because the window is then the study; a narrower
        window is one seek per slot, because the slices of a slot are contiguous and the slots are not.
        """
        with self.path.open("rb", buffering=0) as handle:
            for row in rows:
                base = self.offset + int(batch[row]) * self.per_study
                if span.covers(self.shape[2]):
                    handle.seek(base)
                    handle.readinto(window[row].reshape(-1).data)
                    continue
                for slot in range(self.shape[1]):
                    handle.seek(base + slot * self.strides[1] + span.start * self.per_slot)
                    handle.readinto(window[row, slot].reshape(-1).data)
