from typing import override

import numpy as np
import torch
from torch import Tensor, nn

from kneemri.models.fingerprint import FINGERPRINT, Fingerprint

SEED = 2026
SLOTS = 4
GROUP = 3
SIZE = 8


class _Answering(nn.Module):
    """A stand-in whose answer depends on the pixels AND on the mask, which is what the question asks."""

    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Linear(GROUP * SIZE * SIZE, 3)

    @override
    def forward(self, imgs: Tensor, mask: Tensor, image_size: int) -> Tensor:
        assert image_size == SIZE, "the size is part of the question, so it has to reach the forward pass"
        flat = imgs.float().flatten(start_dim=2) / 255.0
        return (self.weights(flat) * mask.unsqueeze(-1)).mean(dim=1)


def test_of():
    """The question is a constant; the answer is the member's.

    Equivalence classes: the same weights answer identically twice (that is the whole point of a
    seeded bag — nothing about the machine may enter), different weights answer differently by order
    one, and the mask's second branch is exercised because the bag holds two studies, one of them
    masked. The last is what makes the number see a pooling change rather than only a weight change.
    """
    torch.manual_seed(SEED)
    model = _Answering()
    device = torch.device("cpu")

    first = FINGERPRINT.of(model, SLOTS, SIZE, GROUP, device)
    assert first.shape == (FINGERPRINT.batch, 3)
    assert np.array_equal(first, FINGERPRINT.of(model, SLOTS, SIZE, GROUP, device))

    torch.manual_seed(SEED + 1)
    assert not np.allclose(first, FINGERPRINT.of(_Answering(), SLOTS, SIZE, GROUP, device))

    assert not np.allclose(first[0], first[1]), "the masked study must answer differently from the unmasked one"


def test_of_leaves_training_mode_untouched():
    """A check run mid-training must not silently leave the model in eval — every save takes one."""
    torch.manual_seed(SEED)
    model = _Answering()
    model.train()
    FINGERPRINT.of(model, SLOTS, SIZE, GROUP, torch.device("cpu"))
    assert model.training

    model.eval()
    FINGERPRINT.of(model, SLOTS, SIZE, GROUP, torch.device("cpu"))
    assert not model.training


def test_the_question_is_spelled_once():
    """The seed a member STAMPS and the seed it is ASKED are the same object, or the stamp is fiction."""
    assert Fingerprint() == FINGERPRINT, "the module singleton must be the default question, not a tuned one"
    assert FINGERPRINT.batch == 2, "two studies, so the masked branch of the softmax is exercised"
