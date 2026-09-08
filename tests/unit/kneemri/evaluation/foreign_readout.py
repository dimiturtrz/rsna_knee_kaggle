"""The three readings a shift is priced by, and the order they assemble in."""

from pathlib import Path
from typing import override

import numpy as np
import pytest
from numpy.typing import NDArray

from core.data.corpus.study_cache import CachedCorpus
from kneemri.evaluation.foreign_blocks import BLOCKS, MappedSlab
from kneemri.evaluation.foreign_readout import ForeignReadout, ShiftedReading, ShiftPair
from kneemri.evaluation.foreign_slots import SlotsKept
from kneemri.inference.holdout_slab import HoldoutSlab
from kneemri.inference.members import Member

SLOTS = 3
SLICES = 4
SIZE = 2
HIS_SLICES = 8


def _corpus(slots: int, fill: int, slices: int = SLICES) -> CachedCorpus:
    """A tiny corpus whose slot index is written into its own pixels, so a misplacement is visible."""
    images = np.zeros((2, slots, slices, SIZE, SIZE), dtype=np.uint8)
    for slot in range(slots):
        images[:, slot] = fill + slot
    return CachedCorpus(images=images, mask=np.ones((2, slots), dtype=np.float32))


def _ramp() -> CachedCorpus:
    """One his-style volume whose slice INDEX is its pixel value, so order and direction are readable."""
    images = np.zeros((2, 1, HIS_SLICES, SIZE, SIZE), dtype=np.uint8)
    for slice_ in range(HIS_SLICES):
        images[:, 0, slice_] = slice_
    return CachedCorpus(images=images, mask=np.ones((2, 1), dtype=np.float32))


def _member() -> Member:
    """A member built from its record alone — nothing here reads a checkpoint."""
    return Member(
        identifier="m",
        file=Path("member.pt"),
        fold=0,
        config={
            "img": SIZE,
            "slices": SLICES,
            "group": 3,
            "crop_mm": 160.0,
            "band": [0.0, 1.0],
            "slots": [str(slot) for slot in range(SLOTS)],
            "variant": "small",
            "pool": "max",
        },
        fingerprint=np.zeros(1, dtype=np.float32),
        holdout=0.0,
        annot=0.0,
    )


def test_slot_cost() -> None:
    """The plumbing self-check: our own pixels through the shifted path, which must read ~0."""
    assert ShiftedReading("m", 0.8800, 0.8800, 0.8100).slot_cost == pytest.approx(0.0)
    assert ShiftedReading("m", 0.8800, 0.8300, 0.8100).slot_cost == pytest.approx(-0.0500)


def test_decode_cost() -> None:
    """The pixel number: what his span and slice placement cost, once the self-check reads zero."""
    assert ShiftedReading("m", 0.8800, 0.8800, 0.8100).decode_cost == pytest.approx(-0.0700)


def test_drop() -> None:
    """The total a deployment would pay — the two together, and equal to the pixel one when plumbing is clean."""
    reading = ShiftedReading("m", 0.8800, 0.8800, 0.8100)
    assert reading.drop == pytest.approx(-0.0700)
    assert reading.drop == pytest.approx(reading.slot_cost + reading.decode_cost)


def test_logged() -> None:
    """The self-check is printed BESIDE the delta, so a broken path cannot be read as a pixel result."""
    line = ShiftedReading("m", 0.8800, 0.8300, 0.8100).logged()
    assert "slots -0.0500" in line
    assert "decode -0.0200" in line
    assert "total -0.0700" in line


def test_read() -> None:
    """The three numbers assemble in one reading, the native one taken as given rather than recomputed."""

    class Canned(ForeignReadout):
        """A readout whose scoring is answered without a card — the arithmetic is what is under test."""

        @override
        def scored(self, member: Member, slab: HoldoutSlab, truth: NDArray[np.float32], *, overlap: bool) -> float:
            """The self-check reproduces native and the shifted slab does not, so an order swap is visible."""
            return 0.8800 if isinstance(slab, SlotsKept) else 0.8100

    holdout = np.arange(2, dtype=np.int64)
    reading = Canned(
        None,  # pyrefly: ignore[bad-argument-type]
        None,  # pyrefly: ignore[bad-argument-type]
    ).read(
        _member(),
        0.8800,
        ShiftPair(
            SlotsKept(_corpus(SLOTS, fill=10), holdout, kept=(0, 1, 2)),
            MappedSlab(_ramp(), holdout, slots=SLOTS, slices=SLICES, blocks=BLOCKS[:1]),
        ),
        np.zeros((2, 12), dtype=np.float32),
    )

    assert reading.name == "member.pt"
    assert reading.native == pytest.approx(0.8800)
    assert reading.kept == pytest.approx(0.8800)
    assert reading.foreign == pytest.approx(0.8100)
    assert reading.slot_cost == pytest.approx(0.0), "the self-check is clean, so the drop is all decode"
