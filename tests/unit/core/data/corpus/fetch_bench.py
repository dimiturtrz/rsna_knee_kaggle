"""The bench's arithmetic and its control: bytes counted off the array, and arms that cannot overlap."""

import numpy as np
import pytest

from core.data.corpus.fetch_bench import BATCH, BATCHES, FetchBench, Sweep
from core.data.corpus.window_reader import MappedReader, SliceSpan

STUDIES = BATCH * BATCHES * 3
SLOTS, SLICES, PIXELS = 2, 4, 5


def _images() -> np.ndarray:
    return np.zeros((STUDIES, SLOTS, SLICES, PIXELS, PIXELS), dtype=np.uint8)


def test_bytes_read():
    """Counted off the ARRAY's own shape, so a bench pointed at a bigger cache cannot keep quoting the
    byte count of the one it was written against."""
    sweep = Sweep.over(0, STUDIES, SliceSpan(start=0, width=SLICES))
    assert sweep.bytes_read(_images()) == BATCH * BATCHES * SLOTS * SLICES * PIXELS * PIXELS


def test_over():
    """Successive arms get disjoint rows — the control the whole measurement rests on, since an arm
    reading what the previous arm warmed is timing the page cache."""
    first = Sweep.over(0, STUDIES, SliceSpan(start=0, width=SLICES))
    second = Sweep.over(1, STUDIES, SliceSpan(start=0, width=SLICES))
    assert set(np.concatenate(first.rows).tolist()).isdisjoint(np.concatenate(second.rows).tolist())
    assert len(np.concatenate(first.rows)) == BATCH * BATCHES


def test_over_refuses_a_cache_too_small_to_keep_the_arms_apart():
    """Refused rather than wrapped: silently reusing rows would report the last arm as the fastest."""
    with pytest.raises(ValueError, match="arm 2 needs rows"):
        Sweep.over(2, BATCH * BATCHES, SliceSpan(start=0, width=SLICES))


def test_seconds():
    """The whole sweep, not one call — a step pays for every batch it reads."""
    reads: list[int] = []
    sweep = Sweep.over(0, STUDIES, SliceSpan(start=0, width=SLICES))
    images = _images()

    class Counts(MappedReader):
        def read(self, batch, span):  # type: ignore[no-untyped-def]
            reads.append(len(batch))
            return super().read(batch, span)

    assert FetchBench().seconds(Counts(images), sweep) >= 0.0
    assert reads == [BATCH] * BATCHES


def test_rate():
    """GB/s over the bytes the sweep asked for. Positive and finite on a real read, which is what
    separates a working bench from one dividing by a zero it never noticed."""
    images = _images()
    rate = FetchBench().rate(MappedReader(images), Sweep.over(0, STUDIES, SliceSpan(start=0, width=SLICES)), images)
    assert rate > 0.0
    assert np.isfinite(rate)


def test_report():
    """Every arm against the FASTEST, because the reading that matters is which reader to ship."""
    lines = FetchBench().report({"mapped": 1.0, "file(2)": 2.0}, width=2, slices=4)
    assert "2 of 4 slices" in lines[0]
    assert "0.50x" in lines[0]
    assert "1.00x" in lines[1]


def test_report_names_a_whole_stack_read_as_one():
    """A read of every slice is the other branch of `FileReader` — one seek per study, not per slot —
    so the report has to say which branch produced the number."""
    assert "whole stack" in FetchBench().report({"mapped": 1.0}, width=4, slices=4)[0]
