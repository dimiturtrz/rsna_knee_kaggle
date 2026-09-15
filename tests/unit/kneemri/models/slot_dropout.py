"""Slot dropout: pixels and presence move together, absent slots never drop, no study is emptied.

The mirror of `kneemri/models/slot_dropout.py`. Every tensor is built here and seeded.
"""

import torch

from kneemri.models.slot_dropout import SlotDropout

STUDIES, SLOTS = 64, 4


def _bag() -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.full((STUDIES, SLOTS, 1, 2, 2), 7, dtype=torch.uint8)
    presence = torch.ones(STUDIES, SLOTS)
    presence[:, SLOTS - 1] = 0.0
    presence[0, 1:] = 0.0
    return images, presence


def test_dropped():
    """The withheld set is a subset of the acquired one, never all of it, and a rate of 0 draws nothing."""
    images, presence = _bag()
    generator = torch.Generator().manual_seed(0)

    kept_images, kept = SlotDropout(0.9).dropped(images, presence, generator)

    acquired = presence > 0
    assert (kept_images[:, :, 0, 0, 0][acquired] == 7 * kept.to(torch.uint8)[acquired]).all(), "pixels follow presence"
    assert (kept <= presence).all(), "an absent slot is never brought back"
    assert (kept.sum(dim=1) >= 1).all(), "no study is emptied"
    assert kept.sum() < presence.sum(), "at rate 0.9 something is withheld"
    untouched = torch.Generator().manual_seed(0)
    assert SlotDropout(0.0).dropped(images, presence, untouched)[1] is presence
    assert torch.equal(untouched.get_state(), torch.Generator().manual_seed(0).get_state()), "rate 0 draws nothing"
