"""How a batch of studies reaches the trainer: by indexing the cache, or by reading its file outright.

The fetch is synchronous on the training thread, so its cost is wall-clock on every step, and on the
larger caches it is the step's biggest host-side term. Which reader a corpus uses is decided by whether
its pixels are file-backed at all — see `CachedCorpus.reader`.
"""

from concurrent.futures import ThreadPoolExecutor
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
"""


class WindowReader(Protocol):
    """The one thing a reader does: hand back `width` slices of every slot for a batch of positions.

    `(batch, start, width)` travels as three parameters rather than one request object DELIBERATELY. It
    is a slicing expression, not a noun the rest of the codebase says anything about: nothing stores
    one, nothing returns one, and nothing decides anything from one — `Prefetch` already holds the
    schedule that produces them, and `Window` in `inference.members` is a different thing (a member's
    read plan). A value object here would add a type to name an argument list, on the inner loop, and
    would still be unpacked in the same two places.
    """

    def read(self, batch: NDArray[np.int64], start: int, width: int) -> NDArray[np.uint8]: ...


class MappedReader:
    """Reads a batch's window by indexing the array itself — correct for anything, mapped or resident."""

    def __init__(self, images: NDArray[np.uint8]) -> None:
        self.images = images

    def read(self, batch: NDArray[np.int64], start: int, width: int) -> NDArray[np.uint8]:
        """The window, contiguous, in ONE indexing op — never index the batch and slice afterwards."""
        return np.ascontiguousarray(self.images[batch, :, start : start + width])


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
        self.dtype = images.dtype
        self.threads = threads
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="fetch")

    @property
    def per_slot(self) -> int:
        """Bytes in one slot's slice — the unit a window is built from."""
        return int(np.prod(self.shape[3:]))

    @property
    def per_study(self) -> int:
        """Bytes in one study, which is what its position is a multiple of."""
        return int(self.shape[1] * self.shape[2] * self.per_slot)

    def read(self, batch: NDArray[np.int64], start: int, width: int) -> NDArray[np.uint8]:
        """The window for every position in the batch, the rows shared out over the handles."""
        window = np.empty((len(batch), self.shape[1], width, *self.shape[3:]), dtype=self.dtype)
        shares = [range(offset, len(batch), self.threads) for offset in range(self.threads)]
        list(self.pool.map(lambda rows: self._fill(window, batch, start, width, rows), shares))
        return window

    def _fill(
        self,
        window: NDArray[np.uint8],
        batch: NDArray[np.int64],
        start: int,
        width: int,
        rows: range,
    ) -> None:
        """Fill this share of the rows through one handle of its own — handles are not thread-safe.

        A whole-stack read is one seek per study, because the window is then the study; a narrower
        window is one seek per slot, because the slices of a slot are contiguous and the slots are not.
        """
        with self.path.open("rb", buffering=0) as handle:
            for row in rows:
                base = self.offset + int(batch[row]) * self.per_study
                if width == self.shape[2]:
                    handle.seek(base)
                    handle.readinto(window[row].reshape(-1).data)
                    continue
                for slot in range(self.shape[1]):
                    handle.seek(base + (slot * self.shape[2] + start) * self.per_slot)
                    handle.readinto(window[row, slot].reshape(-1).data)
