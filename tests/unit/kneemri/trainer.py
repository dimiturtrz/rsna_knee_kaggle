from dataclasses import dataclass
from typing import override

import numpy as np
import torch
from torch import Tensor

from kneemri.model import TARGETS, StudyModel
from kneemri.study_cache import CachedCorpus
from kneemri.targets import Targets
from kneemri.trainer import Trainer

SEED = 2026
DIM = 8
PATCHES = 16
SLOTS = 6


@dataclass
class _Output:
    last_hidden_state: Tensor


class _FakeBackbone(torch.nn.Module):
    """Stands in for DINOv2, including the `encoder.layer` list the freezing walks."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(DIM, DIM) for _ in range(12)])

    @override
    def forward(self, *, pixel_values: Tensor) -> _Output:
        n = pixel_values.shape[0]
        column = pixel_values.mean(dim=(1, 2, 3)).reshape(n, 1, 1)
        base = torch.arange(PATCHES + 1, dtype=torch.float32).reshape(1, -1, 1)
        return _Output(last_hidden_state=(base + column).expand(n, PATCHES + 1, DIM).contiguous())


def _trainer() -> Trainer:
    torch.manual_seed(SEED)
    return Trainer(StudyModel(_FakeBackbone(), DIM, SLOTS), torch.device("cpu"), unfreeze_last=6)


def test_augment():
    """Jitter moves the pixels but never mirrors them — a flip would undo laterality normalisation."""
    trainer = _trainer()
    generator = torch.Generator().manual_seed(SEED)
    images = torch.rand(4, 3, 16, 16)
    warped = trainer.augment(images, generator)
    assert warped.shape == images.shape
    assert not torch.allclose(warped, images), "augmentation must actually perturb"
    assert float(warped.max()) <= 1.0
    flipped = torch.flip(images, dims=[-1])
    assert not torch.allclose(warped, flipped, atol=0.2), "no horizontal flip may be introduced"


def test_batches():
    """Evaluation keeps corpus order; training shuffles, and both cover every position exactly once."""
    trainer = _trainer()
    generator = np.random.default_rng(SEED)
    positions = np.arange(10, dtype=np.int64)
    ordered = trainer.batches(positions, 4, generator, shuffle=False)
    assert [len(batch) for batch in ordered] == [4, 4, 2]
    assert np.concatenate(ordered).tolist() == positions.tolist()
    shuffled = np.concatenate(trainer.batches(positions, 4, generator, shuffle=True))
    assert sorted(shuffled.tolist()) == positions.tolist()


def test_loss():
    """Weight zero silences a target entirely, which is how an unmentioned finding must behave."""
    trainer = _trainer()
    logits = torch.zeros(2, len(TARGETS))
    scores = torch.ones(2, len(TARGETS))
    weights = torch.zeros(2, len(TARGETS))
    assert float(trainer.loss(logits, scores, weights)) == 0.0
    weights[0, 0] = 1.0
    assert float(trainer.loss(logits, scores, weights)) > 0.0


def test_area_under_curve():
    """A perfect ranking scores 1, a reversed one 0, and a label with one class is undefined."""
    trainer = _trainer()
    truth = np.array([0, 0, 1, 1], dtype=np.int64)
    assert trainer.area_under_curve(np.array([0.1, 0.2, 0.8, 0.9]), truth) == 1.0
    assert trainer.area_under_curve(np.array([0.9, 0.8, 0.2, 0.1]), truth) == 0.0
    assert np.isnan(trainer.area_under_curve(np.array([0.5, 0.6]), np.array([1, 1], dtype=np.int64)))


def test_evaluate():
    """Per-label AUC against the graded targets binarised at their midpoint."""
    trainer = _trainer()
    predictions = np.tile(np.array([[0.1], [0.9]]), (1, len(TARGETS)))
    scores = np.tile(np.array([[0.0], [1.0]], dtype=np.float32), (1, len(TARGETS)))
    per_label = trainer.evaluate(predictions, scores)
    assert set(per_label) == set(TARGETS)
    assert per_label["ACL"] == 1.0


def _corpus(studies: int = 6, slices: int = 6, size: int = 14) -> CachedCorpus:
    generator = np.random.default_rng(SEED)
    images = generator.integers(0, 256, (studies, SLOTS, slices, size, size), dtype=np.uint8)
    mask = np.ones((studies, SLOTS), dtype=np.float32)
    return CachedCorpus(images, mask)


def _targets(studies: int = 6) -> Targets:
    generator = np.random.default_rng(SEED)
    scores = generator.random((studies, len(TARGETS))).astype(np.float32)
    weights = np.ones((studies, len(TARGETS)), dtype=np.float32)
    return Targets(scores, weights, [f"s{index}" for index in range(studies)])


def test_fit_epoch():
    """One pass returns a finite mean loss and actually moves the head's weights."""
    trainer = _trainer()
    before = trainer.model.head.out.weight.detach().clone()
    loss = trainer.fit_epoch(_corpus(), _targets(), np.arange(6, dtype=np.int64), 3, np.random.default_rng(SEED))
    assert np.isfinite(loss)
    assert not torch.allclose(before, trainer.model.head.out.weight), "a fitted epoch must change the head"


def test_predict():
    """Prediction returns one probability per target per study, in [0, 1]."""
    trainer = _trainer()
    predictions = trainer.predict(_corpus(), np.arange(6, dtype=np.int64), 3, np.random.default_rng(SEED))
    assert predictions.shape == (6, len(TARGETS))
    assert predictions.min() >= 0.0
    assert predictions.max() <= 1.0
