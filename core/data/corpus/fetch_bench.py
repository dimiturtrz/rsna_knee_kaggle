"""How fast a batch reaches the trainer, per reader and per handle count:
`python -m core.data.corpus.fetch_bench --slices 12 --image-size 336`.

The fetch is synchronous on the training thread, so its throughput is step wall-clock. `READERS = 2` in
`window_reader` is a legislated constant carrying a measurement: the map served a batch at 0.88 GB/s
where explicit reads served the same bytes at 1.67, and two handles reached 2.27. That measurement was
taken once, by hand, on the 448 px cache and on a WHOLE-STACK read — one seek per study. The trainer's
own read is the other branch: a narrow window is one seek per SLOT, six times the seeks for a fraction
of the bytes, and nothing says the two branches peak at the same handle count.

COMMITTED RATHER THAN RUN ONCE, for the reason `decode_bench` states: a performance claim goes stale
silently, and a constant defended by a number nobody can recompute is a constant defended by nothing.

EACH ARM READS STUDIES NO OTHER ARM READS. The page cache is the whole hazard here — an arm that reads
what the previous arm just read measures RAM, and reports a reader as twice as fast for having run
second. Disjoint row ranges are what make the arms comparable, so the sweep refuses to hand two arms
the same rows.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from core import cli
from core.data.corpus.window_reader import FileReader, MappedReader, WindowReader

logger = logging.getLogger(__name__)

BATCH = 12
"""Studies per read, which is the trainer's own batch — the unit whose cost is a step."""

BATCHES = 8
"""Reads per arm. Enough that one slow seek does not name the arm, small enough that eight arms over
disjoint rows still fit inside a corpus."""

HANDLES = (1, 2, 4, 8)
"""Handle counts to try. `READERS` is the second of these; the fourth is where the drive fell off when
this was last measured, and an arm that is not run cannot say so."""

GIGABYTE = float(1 << 30)
"""Bytes per GB, binary — the unit `np.memmap` sizes are quoted in everywhere else here."""


@dataclass(frozen=True)
class Sweep:
    """One read, repeated over rows nothing else in the run touches.

    The rows are held rather than drawn per arm because DISJOINTNESS IS THE CONTROL: two arms that
    overlap are measuring the page cache, and an arm handed its rows cannot quietly redraw them.
    """

    rows: list[NDArray[np.int64]]
    start: int
    width: int

    def bytes_read(self, images: NDArray[np.uint8]) -> int:
        """Bytes this sweep asks for, from the cache's own shape — never from a size passed alongside."""
        per_slice = int(np.prod(images.shape[3:]))
        return sum(len(batch) * images.shape[1] * self.width * per_slice for batch in self.rows)

    @classmethod
    def over(cls, arm: int, studies: int, start: int, width: int) -> "Sweep":
        """The `arm`-th disjoint block of rows, so arm N never reads what arm N-1 warmed.

        Contiguous rather than strided: the cache is laid out by study, and a strided block would
        measure a seek pattern no reader of this cache performs.
        """
        span = BATCH * BATCHES
        first = arm * span
        if first + span > studies:
            message = f"arm {arm} needs rows {first}..{first + span} of a cache holding {studies}"
            raise ValueError(message)
        rows = [np.arange(first + n * BATCH, first + (n + 1) * BATCH, dtype=np.int64) for n in range(BATCHES)]
        return cls(rows=rows, start=start, width=width)


class FetchBench:
    """Each reader timed on its own rows, reported as the throughput a training step would see."""

    def seconds(self, reader: WindowReader, sweep: Sweep) -> float:
        """Wall-clock for the whole sweep, which is what a step pays — not a per-call best case."""
        started = time.time()
        for batch in sweep.rows:
            reader.read(batch, start=sweep.start, width=sweep.width)
        return time.time() - started

    def rate(self, reader: WindowReader, sweep: Sweep, images: NDArray[np.uint8]) -> float:
        """GB/s: the bytes the sweep asked for over the seconds it took."""
        return sweep.bytes_read(images) / GIGABYTE / max(self.seconds(reader, sweep), 1e-9)

    def report(self, measured: dict[str, float], width: int, slices: int) -> list[str]:
        """Every arm against the fastest, because the question is which reader to ship, not the absolute."""
        best = max(measured.values(), default=1.0) or 1.0
        depth = "whole stack" if width == slices else f"{width} of {slices} slices"
        return [f"{depth}: {label:14s} {rate:5.2f} GB/s   {rate / best:.2f}x" for label, rate in measured.items()]


def main() -> None:
    """Time the real cache under each reader and handle count, on rows no other arm reads."""
    from core.data.corpus.cache_flags import CacheFlags  # noqa: PLC0415 - CLI-only

    command = cli.Command(__doc__, logger)
    CacheFlags.add_to(command.parser)
    command.parser.add_argument("--width", type=int, default=None, help="slices per read (default: the whole stack)")
    command.parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="shift every arm this many blocks along the cache. A rerun at the same offset re-reads rows "
        "the first run warmed, which measures RAM; a rerun at a new one also hands each reader a "
        "different region, which is what separates a reader's rate from where in the file it landed.",
    )
    command.parser.add_argument(
        "--reverse", action="store_true", help="time the readers in the opposite order, against arm-order effects"
    )
    arguments = command.parsed()

    cache = CacheFlags.cache(arguments, command.rooted(arguments).processed())
    corpus, identifiers = cache.load()
    images = corpus.images
    if not isinstance(images, np.memmap) or not images.filename:
        message = f"{cache.key} did not load as a memory map, so there is no file branch to time"
        raise SystemExit(message)
    width = arguments.width or corpus.slices
    logger.info("%s: %d studies, %d slices, %d px", cache.key, len(identifiers), corpus.slices, corpus.image_size)

    bench = FetchBench()
    measured: dict[str, float] = {}
    readers: list[tuple[str, WindowReader]] = [("mapped", MappedReader(images))]
    readers += [(f"file({handles})", FileReader(images, Path(images.filename), handles)) for handles in HANDLES]
    for arm, (label, reader) in enumerate(reversed(readers) if arguments.reverse else readers):
        rows = Sweep.over(arm + arguments.offset, len(identifiers), start=0, width=width)
        measured[label] = bench.rate(reader, rows, images)

    command.report(bench.report(measured, width, corpus.slices))


if __name__ == "__main__":
    main()
