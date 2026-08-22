"""Read the next batch while the current one is on the GPU, instead of in front of it.

The fetch is host work — a file read into a buffer — and the step it precedes is device work, so the two
can run at once and only do not because the loop asks for them in turn. At 448 px whole-stack the read is
about 98 ms that the GPU spends idle every step, which is the largest remaining host-side term after the
reader itself stopped going through the page-fault path (see `window_reader`).

WHAT MAKES THIS SAFE IS THE PLAN. The reader has to run ahead of the loop, so it cannot be the loop that
decides what to read next — the schedule is drawn up front, in step order, and handed over whole. That
also settles determinism rather than hoping for it: the only draws inside the training loop were the
permutation and one window start per step, with nothing between them, so drawing the starts up front in
the same order yields the identical sequence of values. The augmentation keeps its own generator on the
main thread and is untouched, because it belongs to the step, not to the fetch.
"""

from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch
from jaxtyping import UInt8
from numpy.typing import NDArray
from torch import Tensor

from core.data.corpus.study_cache import CachedCorpus


class Prefetch:
    """A schedule of `(positions, window start)` reads, served one step ahead of the consumer.

    One step of lookahead is the whole design: a second would double the resident batch memory to hide a
    latency that one already covers, since the read is shorter than the step it overlaps.
    """

    def __init__(self, corpus: CachedCorpus, plan: list[tuple[NDArray[np.int64], int]], width: int, *, pin: bool):
        self.corpus = corpus
        self.plan = plan
        self.width = width
        self.pin = pin

    def fetch(self, step: int) -> UInt8[Tensor, "b n g h w"]:
        """The pixels for one scheduled step, pinned where a device copy will want them pinned.

        Pinning costs a copy, and it is spent HERE — on the reader's thread, inside the window the step
        is hiding — to buy a `non_blocking` transfer on the main one.
        """
        batch, start = self.plan[step]
        pixels = torch.from_numpy(self.corpus.window(batch, start, self.width))
        return pixels.pin_memory() if self.pin else pixels

    def __iter__(self) -> Iterator[tuple[NDArray[np.int64], UInt8[Tensor, "b n g h w"]]]:
        """Each step's positions with its pixels, the next read already in flight."""
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch") as reader:
            ahead: Future[Tensor] = reader.submit(self.fetch, 0) if self.plan else Future()
            for step in range(len(self.plan)):
                pixels = ahead.result()
                if step + 1 < len(self.plan):
                    ahead = reader.submit(self.fetch, step + 1)
                yield self.plan[step][0], pixels
