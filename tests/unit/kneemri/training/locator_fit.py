"""Locator augmentation, loss and a two-epoch fit, on arrays the test writes."""

import numpy as np
import pytest
import torch

from kneemri.models.locator import CLASSES, Locator, SliceCorpus
from kneemri.training.locator_fit import LocatorAugment, LocatorFit

SIDE = 28
CPU = torch.device("cpu")


torch.manual_seed(0)


def _augment() -> LocatorAugment:
    return LocatorAugment(torch.Generator().manual_seed(0))


def test_uniform() -> None:
    drawn = _augment().uniform(0.4, 1.0, 1000, device=CPU)
    assert float(drawn.min()) >= 0.4 and float(drawn.max()) <= 1.0


def test_geometry() -> None:
    labels = torch.randint(0, 5, (3, SIDE, SIDE))
    moved, drawn = _augment().geometry(torch.rand(3, SIDE, SIDE), labels)
    assert moved.shape == drawn.shape == labels.shape
    assert set(drawn.unique().tolist()) <= set(range(5))


def test_intensity() -> None:
    unit = _augment().intensity(torch.rand(4, SIDE, SIDE))
    assert float(unit.min()) >= 0 and float(unit.max()) <= 1


def test_call() -> None:
    gray = torch.full((2, SIDE, SIDE), 128, dtype=torch.uint8)
    slices, labels = _augment()(gray, torch.ones(2, SIDE, SIDE, dtype=torch.long))
    assert float(slices.max()) <= 255
    assert labels.dtype == torch.long


@pytest.mark.parametrize(("right", "lower"), [(True, True), (False, False)])
def test_loss(*, right: bool, lower: bool) -> None:
    labels = torch.randint(0, len(CLASSES), (2, 8, 8))
    logits = torch.nn.functional.one_hot(labels, len(CLASSES)).permute(0, 3, 1, 2).float() * 20
    wrong = torch.nn.functional.one_hot((labels + 1) % len(CLASSES), len(CLASSES)).permute(0, 3, 1, 2).float() * 20
    assert (LocatorFit.loss(logits if right else wrong, labels) < 1) == lower


def test_dice(slice_corpus: SliceCorpus, tiny_locator: Locator) -> None:
    scores = LocatorFit(tiny_locator, slice_corpus, CPU, torch.Generator().manual_seed(0)).dice("test")
    assert set(scores) == set(CLASSES[1:])
    assert all(0 <= score <= 1 for score in scores.values())


def test_fit(slice_corpus: SliceCorpus, tiny_locator: Locator) -> None:
    before = [p.detach().clone() for p in tiny_locator.head.parameters()]
    loss = LocatorFit(tiny_locator, slice_corpus, CPU, torch.Generator().manual_seed(0)).fit(2, batch=2)
    assert np.isfinite(loss)
    assert any(not torch.equal(b, a) for b, a in zip(before, tiny_locator.head.parameters(), strict=True))
