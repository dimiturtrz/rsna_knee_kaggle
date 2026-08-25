"""The blend that makes a memorised study-to-label pair wrong, and the null policy that turns it off."""

import numpy as np
import pytest
import torch

from kneemri.training.mixup import OFF, UNMIXED, Blend, Mixup

SEED = 2026


def a_blend(weight: float) -> Blend:
    """Two rows swapped, so every method can be checked against a hand-computed partner."""
    return Blend(weight, np.array([1, 0], dtype=np.int64))


def test_identity():
    """A full-weight draw is a no-op, and that is what the null policy returns."""
    assert a_blend(UNMIXED).identity()
    assert not a_blend(0.7).identity()
    assert Mixup().draw(4, np.random.default_rng(SEED)).identity(), "alpha 0 must draw no blend at all"


def test_images():
    """Pixels blend in float and come back ROUNDED, or every blended pixel is biased half a level down.

    A plain cast truncates: 0.75*200 + 0.25*100 = 175.0 is safe, but 0.75*201 + 0.25*100 = 175.75 casts
    to 175 and rounds to 176. Small per pixel, systematic across the corpus, and it would read as a
    slightly worse recipe rather than as a defect.
    """
    images = torch.tensor([[[[[201]]]], [[[[100]]]]], dtype=torch.uint8)
    blended = a_blend(0.75).images(images)
    assert blended.dtype == torch.uint8, "the model's forward reads uint8 and must keep reading it"
    assert int(blended[0]) == 176, "0.75*201 + 0.25*100 = 175.75 rounds up, it does not truncate"
    assert int(blended[1]) == 125
    assert torch.equal(a_blend(UNMIXED).images(images), images), "the identity blend returns the input"


def test_mask():
    """Slot presence is the UNION, not the weighted mean — a slot with pixels from either study has pixels.

    Averaging would hand the model a half-present slot no study in the corpus has, and would make the
    mask's meaning depend on the draw.
    """
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert torch.equal(a_blend(0.75).mask(mask), torch.ones(2, 2))
    assert torch.equal(a_blend(UNMIXED).mask(mask), mask)


def test_rows():
    """Targets move WITH the pixels — this is the whole mechanism, not a detail of it.

    Blending pixels alone is an augmentation the memorised answer still satisfies. Blending the target
    by the same weight and the same order is what makes that answer wrong.
    """
    rows = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    blended = a_blend(0.75).rows(rows)
    assert blended.dtype == np.float32, "the loss reads float32 and a promotion here would reach it"
    assert blended[0] == pytest.approx([0.75, 0.25])
    assert blended[1] == pytest.approx([0.25, 0.75])
    assert np.array_equal(a_blend(UNMIXED).rows(rows), rows)


def test_enabled():
    """Off is off. The witness asks this rather than the trainer's loop, so the flag has a positive report."""
    assert not Mixup().enabled(), "the default must be the loop every ledger number was measured under"
    assert not Mixup(OFF).enabled()
    assert Mixup(0.2).enabled()


def test_draw():
    """A draw folds to the dominant half and permutes the batch, from the epoch's own generator.

    FOLDED because the blend is symmetric — weight w against a permutation is the same family of
    examples as 1-w — so the fold costs no diversity and keeps every row recognisably its own study.
    SEEDED because a run that cannot be repeated cannot be compared to its own control.
    """
    drawn = [Mixup(0.4).draw(6, np.random.default_rng(seed)) for seed in (SEED, SEED, SEED + 1)]
    assert all(blend.weight >= UNMIXED / 2 for blend in drawn), "a draw below half is folded, never used as-is"
    assert all(sorted(blend.order.tolist()) == list(range(6)) for blend in drawn), "the order is a permutation"
    assert drawn[0].weight == drawn[1].weight, "one seed, one draw"
    assert np.array_equal(drawn[0].order, drawn[1].order)
    assert not np.array_equal(drawn[0].order, drawn[2].order), "a different seed must move the partner pool"
    identity = Mixup().draw(3, np.random.default_rng(SEED))
    assert identity.order.tolist() == [0, 1, 2], "the null policy pairs every row with itself"
