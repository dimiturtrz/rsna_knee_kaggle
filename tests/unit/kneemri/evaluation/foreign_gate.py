"""A gate whose feed is known to be perfect, and one whose feed is known to be inverted."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.findings import TARGETS
from kneemri.evaluation.foreign_gate import FeedGate, GateReading
from kneemri.inference.foreign_member import ForeignMember, ForeignSpec, WindowFeed

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


def _member() -> ForeignMember:
    """A rebuilt member with the published label order, untrained — its numbers are not the subject here."""
    return ForeignMember(ForeignSpec(ARCH, RES, tuple(TARGETS), GOLD, DIM)).eval()


def _annotations() -> np.ndarray:
    """Two positives and two negatives on every finding, so no column is degenerate."""
    return np.array([[float(row % 2) for _ in TARGETS] for row in range(STUDIES)], dtype=np.float32)


def test_macro():
    """The macro is the unweighted mean of the per-finding AUCs, which is what `gold_auc` also is."""
    reading = GateReading(np.full(len(TARGETS), 0.8), GOLD)

    assert reading.macro == pytest.approx(0.8)


def test_shortfall():
    """A feed that loses ground against the published number reports a negative shortfall."""
    assert GateReading(np.full(len(TARGETS), 0.8), GOLD).shortfall == pytest.approx(0.8 - GOLD)
    assert GateReading(np.full(len(TARGETS), GOLD), GOLD).shortfall == pytest.approx(0.0)


def test_lines():
    """Every finding gets its own row, so one collapsed label is visible rather than averaged away."""
    lines = GateReading(np.full(len(TARGETS), 0.8), GOLD).lines("title")

    assert lines[0] == "title"
    assert sum(target in line for line in lines for target in TARGETS) == len(TARGETS)
    assert "0.8000" in lines[-1] and f"{GOLD:.4f}" in lines[-1]


def test_reading(tmp_path: Path):
    """A gate scores one AUC per finding for the rows it is given, all of them in range."""
    gate = FeedGate(_member(), _feed(tmp_path / "cache"), torch.device("cpu"), GOLD)

    reading = gate.reading([0, 1, 2, 3], _annotations())

    assert reading.aucs.shape == (len(TARGETS),)
    assert ((reading.aucs >= 0.0) & (reading.aucs <= 1.0)).all()
    assert reading.gold == pytest.approx(GOLD), "the oracle is the checkpoint's own number, never a constant here"
