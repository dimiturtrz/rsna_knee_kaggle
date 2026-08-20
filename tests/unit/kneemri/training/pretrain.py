from pathlib import Path
from typing import override

import numpy as np
import torch
from torch import nn

from kneemri.corpus.study_cache import CachedCorpus
from kneemri.training.pretrain import PRESENT, Optimisation, Pretrainer, Sampling
from kneemri.training.spectral_mae import SpectralMae, SpectralMaskConfig

PATCH = 4
GRID = 3
SIDE = PATCH * GRID
TOKENS = GRID * GRID
DIM = 8
STUDIES = 4
SLOTS = 3
SLICES = 5
PER_STEP = 6


class _Backbone(nn.Module):
    """The same stand-in the objective's own tests use: a unit test may not reach the network."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(SIDE * SIDE, DIM)

    @override
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> object:
        token = self.projection(pixel_values.mean(dim=1).flatten(1)).unsqueeze(1)
        return type("Output", (), {"last_hidden_state": token.expand(-1, 1 + TOKENS, DIM).contiguous()})


def _corpus(absent: tuple[int, int] | None = None) -> CachedCorpus:
    """A cache whose pixel VALUE encodes its own address, so a mis-indexed read is visible in the batch."""
    images = np.zeros((STUDIES, SLOTS, SLICES, SIDE, SIDE), dtype=np.uint8)
    for study in range(STUDIES):
        for slot in range(SLOTS):
            images[study, slot] = study * SLOTS + slot + 1
    mask = np.ones((STUDIES, SLOTS), dtype=np.float32)
    if absent is not None:
        mask[absent] = 0.0
    return CachedCorpus(images=images, mask=mask)


def _trainer(corpus: CachedCorpus | None = None, per_step: int = PER_STEP, max_steps: int | None = None) -> Pretrainer:
    objective = SpectralMae(_Backbone(), DIM, SpectralMaskConfig(patch=PATCH))
    sampling = Sampling(per_step, torch.Generator().manual_seed(7), max_steps)
    return Pretrainer(objective, corpus or _corpus(), torch.device("cpu"), sampling)


def test_steps_per_epoch():
    """One pass over every cached slice, and never zero steps however small the corpus is."""
    assert _trainer().steps_per_epoch() == STUDIES * SLOTS * SLICES // PER_STEP
    huge = _trainer(per_step=STUDIES * SLOTS * SLICES * 10)
    assert huge.steps_per_epoch() == 1, "a step larger than the corpus is still one step, not none"

    assert _trainer(max_steps=2).steps_per_epoch() == 2, "the smoke cap must truncate the epoch"


def test_batch():
    """Correct shape and range, and ABSENT slots are never read.

    The absent-slot half is the one worth a test. A sampler that drew studies and then discarded their
    empty slots would silently reweight the corpus toward studies that acquired every series — and the
    third slot of a scheme is missing far more often than the first, so the bias is not uniform.
    """
    corpus = _corpus(absent=(0, 0))
    images = _trainer(corpus).batch()
    assert images.shape == (PER_STEP, 1, SIDE, SIDE)
    assert images.dtype == torch.float32
    assert images.min() >= 0.0 and images.max() <= 1.0, "uint8 must arrive scaled into [0, 1]"

    trainer = _trainer(corpus)
    drawn = torch.cat([trainer.batch().flatten() for _ in range(20)])
    assert not torch.isclose(drawn, torch.tensor(1.0 / 255.0)).any(), "slot (0, 0) is absent and was read"
    assert (corpus.mask >= PRESENT).sum() == STUDIES * SLOTS - 1


def test_one_cycle():
    """The pair is built together, and the schedule is driving the optimiser it was built from."""
    objective = SpectralMae(_Backbone(), DIM, SpectralMaskConfig(patch=PATCH))
    optimisation = Optimisation.one_cycle(objective.parameters(), lr=1.0e-3, steps=4)
    assert optimisation.schedule.optimizer is optimisation.optimiser, "a schedule stepping a stranger"
    assert optimisation.optimiser.param_groups[0]["lr"] < 1.0e-3, "one-cycle starts below its peak"


def test_apply():
    """A backward pass reaches the weights AND advances the rate, in that order."""
    objective = SpectralMae(_Backbone(), DIM, SpectralMaskConfig(patch=PATCH))
    optimisation = Optimisation.one_cycle(objective.parameters(), lr=1.0e-3, steps=4)
    before = objective.decoder.weight.detach().clone()
    rate = optimisation.optimiser.param_groups[0]["lr"]
    optimisation.apply((objective.decoder.weight**2).sum())
    assert not torch.equal(before, objective.decoder.weight), "the gradient never reached the decoder"
    assert optimisation.optimiser.param_groups[0]["lr"] != rate, "the schedule never advanced"


def test_step():
    """One step moves the weights and reports both loss terms."""
    trainer = _trainer()
    before = trainer.objective.decoder.weight.detach().clone()
    optimisation = Optimisation.one_cycle(trainer.objective.parameters(), lr=1.0e-3, steps=2)
    pixels, spectral = trainer.step(optimisation)
    assert pixels > 0.0 and spectral > 0.0
    assert not torch.equal(before, trainer.objective.decoder.weight), "the gradient never reached the decoder"


def test_run():
    """A short run completes every step of every epoch and leaves the model changed."""
    trainer = _trainer(per_step=STUDIES * SLOTS * SLICES)
    optimisation = Optimisation.one_cycle(trainer.objective.parameters(), lr=1.0e-3, steps=2)
    before = trainer.objective.decoder.weight.detach().clone()
    trainer.run(2, optimisation)
    assert not torch.equal(before, trainer.objective.decoder.weight)


def test_save(tmp_path: Path):
    """The recipe travels with the weights, because the write-up's first question is whether it landed."""
    destination = tmp_path / "adapted.pt"
    _trainer().save(destination, "small", epochs=3)
    saved = torch.load(destination, map_location="cpu", weights_only=True)
    assert saved["variant"] == "small"
    assert saved["config"]["epochs"] == 3
    assert saved["config"]["spectral_weight"] == SpectralMaskConfig.spectral_weight
    assert saved["encoder"], "an empty encoder is the silent failure this file exists to make loud"
