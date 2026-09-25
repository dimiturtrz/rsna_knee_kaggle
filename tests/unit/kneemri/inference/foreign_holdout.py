"""A borrowed study list, a cache that holds it, and a cache that quietly does not."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.findings import TARGETS
from kneemri.inference.foreign_holdout import ForeignHoldout
from kneemri.inference.foreign_member import ForeignMember, ForeignSpec, WindowFeed
from kneemri.runs.holdout_record import DERIVED, TRAINED, UNKNOWN_EPOCH, HoldoutReading, HoldoutRecord

ARCH = "resnet18"
DIM = 512
RES = 32
SLOTS = 2
SLICES = 4
SIZE = 16
STUDIES = 4
GOLD = 0.9167


def _feed(path: Path) -> WindowFeed:
    """Four studies whose pixels are their own row index, so a misread row changes the answer."""
    path.mkdir(parents=True, exist_ok=True)
    images = np.zeros((STUDIES, SLOTS, SLICES, SIZE, SIZE), dtype=np.uint8)
    for row in range(STUDIES):
        images[row] = row + 1
    np.save(path / "images_k.npy", images)
    np.save(path / "mask_k.npy", np.ones((STUDIES, SLOTS), dtype=np.float32))
    (path / "index_k.json").write_text(json.dumps([f"s{row}" for row in range(STUDIES)]), encoding="utf-8")
    return WindowFeed.of(path, "k", RES)


def _borrowed() -> HoldoutReading:
    """A member's record over two of the four studies, out of cache order so the join must be by name."""
    studies = ["s2", "s0"]
    return HoldoutReading(
        studies=studies,
        predictions=np.zeros((len(studies), len(TARGETS)), dtype=np.float64),
        truth=np.array([[float(row % 2) for _ in TARGETS] for row in range(len(studies))], dtype=np.float32),
        epoch=3,
        source=TRAINED,
    )


def _holdout(path: Path) -> ForeignHoldout:
    member = ForeignMember(ForeignSpec(ARCH, RES, tuple(TARGETS), GOLD, DIM)).eval()
    return ForeignHoldout(member, _feed(path), torch.device("cpu"))


def test_rows():
    """Studies map to their own cache positions, and one the cache lacks is refused rather than dropped."""
    order = [f"s{row}" for row in range(STUDIES)]

    assert ForeignHoldout.rows(order, ["s2", "s0"]).tolist() == [2, 0]

    with pytest.raises(ValueError, match="absent from this cache"):
        ForeignHoldout.rows(order, ["s0", "s9"])


def test_reading(tmp_path: Path):
    """The foreign reading carries the borrowed studies and the borrowed truth, stamped as re-derived."""
    holdout = _holdout(tmp_path / "cache")
    adopted = _borrowed()

    reading = holdout.reading(adopted, [f"s{row}" for row in range(STUDIES)])

    assert reading.studies == adopted.studies
    assert np.array_equal(reading.truth, adopted.truth), "the ruler is borrowed whole, never recomputed"
    assert reading.predictions.shape == (len(adopted.studies), len(TARGETS))
    assert ((reading.predictions >= 0.0) & (reading.predictions <= 1.0)).all()
    assert reading.source == DERIVED
    assert reading.epoch == UNKNOWN_EPOCH, "foreign weights have no epoch of ours to name"


def test_borrowed(tmp_path: Path):
    """The record is read from the named member, and a member without one is refused by name."""
    adopted = _borrowed()
    HoldoutRecord(tmp_path / "mine.pt").write(
        adopted.studies, adopted.predictions, adopted.truth, adopted.epoch, source=adopted.source
    )

    assert ForeignHoldout.borrowed(tmp_path, "mine").studies == adopted.studies

    with pytest.raises(FileNotFoundError, match="no holdout record"):
        ForeignHoldout.borrowed(tmp_path, "absent")
