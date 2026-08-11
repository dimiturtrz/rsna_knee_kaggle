from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import torch
from torch import Tensor

from kneemri.model import TARGETS, StudyModel
from kneemri.study_cache import CachedCorpus
from kneemri.targets import Targets
from kneemri.trainer import AugmentConfig, Trainer

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


def test_augment_bag():
    """Bags stay uint8 and keep their shape, and every slot is perturbed rather than passed through.

    This exists because `augment` was written, unit-tested and then never called by `fit_epoch` — the
    training loop ran with no augmentation at all, which the dead-code gate could not see (the method
    WAS referenced: by its test) and the mirror gate could not see (it HAD a test). A test on the caller
    is what catches that class of bug.
    """
    trainer = _trainer()
    torch.manual_seed(SEED)
    bag = torch.randint(0, 256, (2, SLOTS, 3, 16, 16), dtype=torch.uint8)
    out = trainer.augment_bag(bag)
    assert out.shape == bag.shape
    assert out.dtype == torch.uint8
    assert not torch.equal(out, bag), "augmentation must actually perturb the bag"


def test_fit_epoch_augments():
    """`fit_epoch` must route its pixels through augmentation — the bug was that it did not."""
    trainer = _trainer()
    seen: list[torch.Tensor] = []
    original = trainer.augment_bag
    trainer.augment_bag = lambda images: seen.append(images) or original(images)  # type: ignore[method-assign]
    trainer.fit_epoch(_corpus(), _targets(), np.arange(6, dtype=np.int64), 3, np.random.default_rng(SEED))
    assert seen, "fit_epoch called no augmentation at all"
    assert seen[0].dtype == torch.uint8


def test_one_cycle():
    """The cycle starts BELOW the peak, warms up to it, and decays far below it by the end.

    Comparing against the pre-attach rate would prove nothing: `max_lr` IS the constant rate, so a
    warmed-up schedule passes through exactly the value it started from. The contract is the SHAPE.
    """
    trainer = _trainer()
    peak = [group["lr"] for group in trainer.optimiser.param_groups]
    total_steps = 40
    trainer.one_cycle(total_steps=total_steps)
    assert trainer.schedule is not None
    warmup_start = [group["lr"] for group in trainer.optimiser.param_groups]
    assert all(start < top for start, top in zip(warmup_start, peak, strict=True)), "must warm UP to the peak"

    for _ in range(total_steps):
        trainer.schedule.step()
    ended = [group["lr"] for group in trainer.optimiser.param_groups]
    assert all(end < top / 100 for end, top in zip(ended, peak, strict=True)), "must decay far below the peak"


def test_geometric_only():
    """The default must reproduce the pre-existing recipe exactly, or it silently changes members.

    Folds trained before the intensity block existed have to remain comparable with folds trained
    after it, so `geometric_only` is asserted to leave the intensity distribution alone: every pixel
    of an image scales by the SAME factor, which is what a gain is and a gamma/contrast/blur is not.
    """
    trainer = _trainer()
    trainer.augment_config = AugmentConfig.geometric_only()
    images = torch.rand(4, 3, 16, 16) * 0.8 + 0.1
    generator = torch.Generator().manual_seed(7)
    out = trainer._intensity(images, generator)
    ratio = out / images
    assert float(ratio.flatten(1).std(1).max()) < 1e-3, "a pure gain scales every pixel identically"


def test_intensity_perturbs_and_stays_in_range():
    """The full config must actually bite, and must not leave [0, 1] — the cache is uint8."""
    trainer = _trainer()
    trainer.augment_config = AugmentConfig()
    images = torch.rand(8, 3, 16, 16)
    out = trainer._intensity(images, torch.Generator().manual_seed(7))
    assert float((out - images).abs().gt(1e-4).float().mean()) > 0.5, "intensity augmentation must perturb"
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_loss_normalisation_modes_differ():
    """By-weight and by-count are different scales, which is why they are measured as a pair."""
    logits = torch.zeros(2, len(TARGETS))
    scores = torch.ones(2, len(TARGETS))
    weights = torch.full((2, len(TARGETS)), 0.25)
    by_weight = _trainer().loss(logits, scores, weights)
    counted = _trainer()
    counted.normalise_by_count = True
    assert float(counted.loss(logits, scores, weights)) < float(by_weight)


def test_save_partial(tmp_path: Path):
    """The best weights hit disk the moment they improve, and a partial write cannot survive."""
    trainer = _trainer()
    destination = tmp_path / "member_fold0.pt"
    state = {key: value.detach().cpu() for key, value in trainer.model.head.state_dict().items()}
    trainer.save_partial(destination, state, holdout=0.81, epoch=7)

    partial = destination.with_suffix(".partial")
    assert partial.exists()
    assert not partial.with_suffix(".tmp").exists(), "the temp file must be renamed, not left behind"
    reloaded = torch.load(partial, map_location="cpu", weights_only=True)
    assert reloaded["holdout"] == 0.81
    assert reloaded["epoch"] == 7
    assert reloaded["model"].keys() == state.keys()
