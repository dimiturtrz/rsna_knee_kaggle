"""Members, records and unreadable slabs — the three fixtures every reading test in this package builds.

Shared rather than copied because `holdout_predictions`, `trainer_record` and `prediction_cache` now test
three halves of one path: the store, the substitution, and the caller that orders them. Each needs the
same member on disk, and a member's config is twenty lines of dictionary.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from core.data.corpus.study_cache import CachedCorpus
from kneemri.inference.holdout_slab import HoldoutSlab
from kneemri.inference.members import Member
from kneemri.runs.holdout_record import TRAINED, HoldoutRecord

ROWS = np.array([0, 1])
"""Rows the trap corpus does not hold, so any read of it raises instead of quietly returning."""


@pytest.fixture
def reading() -> Callable[[Path], Member]:
    """A member whose checkpoint exists on disk, because the cache key stats it."""

    def make(directory: Path) -> Member:
        checkpoint = directory / "m_x.pt"
        checkpoint.write_bytes(b"weights")
        return Member(
            identifier="x",
            file=checkpoint,
            fold=0,
            config={
                "img": 336,
                "slices": 12,
                "group": 3,
                "crop_mm": 130.0,
                "band": [0.2, 0.8],
                "slots": ["a", "b"],
                "variant": "small",
                "pool": "cls_mean",
            },
            fingerprint=np.zeros((2, 12), np.float32),
            holdout=0.8,
            annot=0.8,
        )

    return make


@pytest.fixture
def recorded(reading: Callable[[Path], Member]) -> Callable[..., Member]:
    """A member whose run left a holdout record, which is the substitution's whole precondition."""

    def make(directory: Path, studies: list[str], source: str = TRAINED) -> Member:
        member = reading(directory)
        predictions = np.arange(len(studies) * 2, dtype=np.float64).reshape(len(studies), 2) / 10.0
        HoldoutRecord(member.file).write(
            studies, predictions, np.zeros((len(studies), 2), np.float32), epoch=3, source=source
        )
        return member

    return make


@pytest.fixture
def unreadable_slab() -> Callable[[int], HoldoutSlab]:
    """A slab that cannot be READ but knows its depth, so a substitution is proven to touch no pixels."""

    def make(slices: int) -> HoldoutSlab:
        corpus = CachedCorpus(np.zeros((0, 1, slices, 1, 1), np.uint8), np.zeros((0, 1), np.float32))
        return HoldoutSlab(corpus, ROWS)

    return make
