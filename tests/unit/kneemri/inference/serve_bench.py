"""The bench's arithmetic and its synthetic workload, on a card-free device."""

import time
from pathlib import Path

import numpy as np
import torch

from kneemri.inference.members import Member
from kneemri.inference.serve_bench import ServeBench

SLOW = 0.020
"""Seconds of busy-work; only the ORDER of the timings matters to these tests."""

SLOTS = ["a", "b"]
SLICES = 4
SIZE = 8
"""A member's shape, small enough to allocate on a CPU and still exercise all five dimensions."""


def bench(read_size: int = 0) -> ServeBench:
    """The harness under test, on a real member and pinned to the CPU so no test needs a free card."""
    member = Member(
        identifier="bench",
        file=Path("bench.pt"),
        fold=0,
        config={
            "img": SIZE,
            "slices": SLICES,
            "group": 1,
            "crop_mm": 130.0,
            "band": [0.2, 0.8],
            "slots": SLOTS,
            "variant": "small",
            "pool": "cls_mean",
        },
        fingerprint=np.zeros((1, 12), np.float32),
        holdout=0.8,
        annot=0.8,
    )
    return ServeBench(torch.device("cpu"), member.resized(read_size) if read_size else member)


def test_pixels():
    """Synthetic uint8 of exactly the rank and dtype a real read has, and an all-present mask."""
    images, mask = bench().pixels(studies=3)

    assert images.shape == (3, len(SLOTS), SLICES, SIZE, SIZE), "studies x slots x slices x h x w"
    assert images.dtype == torch.uint8, "the kernel sends bytes; a float read would time a different forward"
    assert mask.shape == (3, len(SLOTS))
    assert bool((mask == 1).all()), "a dropped slot would make the two arms forward different work"


def test_pixels_for_a_resized_member():
    """A resized member allocates the bigger tensor, which is half of what makes the price real.

    The other half is `window(...).size` moving with it — see `test_resized` beside the member, since
    a tensor the model resamples away times the member's OWN resolution and looks like no change.
    """
    images, _ = bench(read_size=SIZE * 2).pixels(studies=2)

    assert images.shape[-2:] == (SIZE * 2, SIZE * 2), "the resized member is what gets allocated and forwarded"
    assert bench().pixels(studies=2)[0].shape[-1] == SIZE, "and an un-resized member keeps its own size"


def test_timed():
    """The first call is a warm-up and never the reading — a compiled arm spends seconds in it."""
    calls: list[int] = []

    def run() -> object:
        calls.append(len(calls))
        return None

    seconds = bench().timed(run, repeats=3)

    assert len(calls) == 4, "one warm-up plus one call per repeat"
    assert seconds >= 0.0


def test_timed_takes_the_minimum():
    """Noise on a shared card only ADDS time, so the fastest pass is the least contaminated one."""
    delays = iter([0.0, SLOW, 0.0])

    def run() -> object:
        deadline = next(delays, 0.0)
        started = time.perf_counter()
        while time.perf_counter() - started < deadline:
            pass
        return None

    assert bench().timed(run, repeats=2) < SLOW, "the slow pass is not what gets reported"


def test_synchronise():
    """On a device with no queue there is nothing to wait for, and asking torch would raise."""
    assert bench().synchronise() is None


def test_graphs():
    """Readable without a compile having happened — an uncompiled process has built none."""
    assert bench().graphs() >= 0


def test_report():
    """Both arms, what compilation cost, and how many passes pay that cost back."""
    lines = bench().report(eager=0.100, compiled=0.080, warmup=2.0, graphs=1)

    assert "100.0 ms/pass" in lines[0]
    assert "-20%" in lines[1], "a fifth off the pass is what the arm bought"
    assert "pays back after 100 passes" in lines[2], "2 s of compile against 20 ms saved a pass"
    assert "graphs" in lines[3]


def test_report_when_compiling_loses():
    """A compiled arm that is SLOWER never pays back, and the report says so rather than dividing."""
    lines = bench().report(eager=0.080, compiled=0.100, warmup=2.0, graphs=3)

    assert "+25%" in lines[1]
    assert "inf passes" in lines[2]
