"""Slot-dropout TTA: the mask is the training rule, the average is in probability space, plain is checked.

The mirror of `kneemri/evaluation/slotdrop_tta.py`. Every array is built here and seeded; no model, no
cache and no record on disk, because the subjects are the mask contract and the arithmetic over reads.
"""

import numpy as np
import pytest

from core.data.corpus.study_cache import CachedCorpus
from kneemri.evaluation.slotdrop_tta import SlotDropReading, SlotDropSlab
from kneemri.runs.holdout_record import HoldoutReading

STUDIES, SLOTS = 16, 4
LABELS = 12
TRUTH = np.tile(np.array([[0.0], [1.0], [0.0], [1.0]], np.float32), LABELS)


def _corpus() -> CachedCorpus:
    images = np.full((STUDIES, SLOTS, 1, 1, 1), 7, np.uint8)
    mask = np.ones((STUDIES, SLOTS), np.float32)
    mask[:, SLOTS - 1] = 0.0
    mask[0, 1:] = 0.0
    return CachedCorpus(images, mask)


def _stored(predictions: np.ndarray) -> HoldoutReading:
    return HoldoutReading(["s0", "s1", "s2", "s3"], predictions.astype(np.float64), TRUTH, epoch=3)


def test_batches():
    """Pixels and presence drop together, absent slots never drop, no study empties, and a seed repeats."""
    holdout = np.arange(STUDIES, dtype=np.int64)
    first = list(SlotDropSlab(_corpus(), holdout, rate=0.9, seed=1).batches(8))
    again = list(SlotDropSlab(_corpus(), holdout, rate=0.9, seed=1).batches(8))
    plain = list(SlotDropSlab(_corpus(), holdout, rate=0.0, seed=1).batches(8))

    images = np.concatenate([batch[0] for batch in first])
    presence = np.concatenate([batch[1] for batch in first])
    acquired = _corpus().mask > 0
    assert (images[:, :, 0, 0, 0][acquired] == 7 * presence[acquired]).all(), "withheld: zero pixels AND presence"
    assert (presence[:, SLOTS - 1] == 0).all() and (images[:, SLOTS - 1] == 7).all(), "absent is left as served"
    assert (presence.sum(axis=1) >= 1).all(), "no study is emptied"
    assert presence.sum() < _corpus().mask.sum(), "at rate 0.9 something is withheld"
    assert all((a[1] == b[1]).all() for a, b in zip(first, again, strict=True)), "one seed, one mask"
    assert (np.concatenate([batch[1] for batch in plain]) == _corpus().mask).all(), "rate 0 is the identity"


def test_positions():
    """The record's order is kept, and a study the cache does not hold is refused rather than dropped."""
    assert SlotDropReading.positions(["a", "b", "c"], ["c", "a"]).tolist() == [2, 0]
    with pytest.raises(ValueError, match="not in the member's cache"):
        SlotDropReading.positions(["a", "b"], ["a", "x"])


def test_averaged():
    """The mean is over draws in probability space, not a rank vote."""
    draws = (np.array([[0.1, 0.9]]), np.array([[0.3, 0.5]]))
    reading = SlotDropReading(_stored(TRUTH), np.array([[0.2, 0.7]]), draws)

    assert reading.averaged() == pytest.approx(np.array([[0.2, 0.7]]))


def test_drift():
    """Measured over the rows scored, so a `--studies` prefix is compared with the record's prefix."""
    stored = _stored(np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5], [0.5, 0.5]]))
    exact = SlotDropReading(stored, stored.predictions[:2].copy(), ())
    off = SlotDropReading(stored, stored.predictions[:2] + 0.01, ())

    assert exact.drift() == 0.0
    assert off.drift() == pytest.approx(0.01)


def test_lines():
    """Stored, plain and the dropout mean under both rulers, and a verdict that names a mismatch."""
    perfect = np.tile(np.array([[0.1], [0.9], [0.2], [0.8]]), LABELS)
    reading = SlotDropReading(_stored(perfect), perfect + 0.001, (1.0 - perfect, 1.0 - perfect))

    lines = reading.lines()

    assert lines[1].split() == ["stored", "1.0000", "1.0000"]
    assert lines[3].split()[-2:] == ["0.0000", "0.0000"], "the inverted draws score as inverted"
    assert "DIFFERS FROM" in lines[4], "a gap above tolerance is said, not hidden"
