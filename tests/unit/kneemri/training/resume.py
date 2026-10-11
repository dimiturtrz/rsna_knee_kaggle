"""A trajectory survives the write, and putting it back continues the same schedule.

The trainer is stood in for by the five attributes `ResumeState` actually reads — an optimiser, a
scheduler, a gradient scaler and two generators. A real `Trainer` needs a model, a device and a mapped
corpus, none of which this asks a question about, and a unit test may touch nothing it did not create.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import torch

from kneemri.runs.artefact import Artefact
from kneemri.training.resume import ResumeState
from kneemri.training.trainer import Trainer

SEED = 2026
STEPS = 10


def _trainer() -> tuple[Trainer, torch.optim.lr_scheduler.OneCycleLR]:
    """An object carrying exactly the surface a trajectory is taken from and put back onto.

    The schedule travels beside it because `Trainer.schedule` is optional and this stand-in always has
    one — a test that asserts on the curve's position may not first have to prove it exists.
    """
    parameter = torch.nn.Parameter(torch.ones(4))
    optimiser = torch.optim.AdamW([parameter], lr=0.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=0.01, total_steps=STEPS)
    trainer = SimpleNamespace(
        optimiser=optimiser,
        schedule=schedule,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        torch_generator=torch.Generator().manual_seed(SEED),
    )
    return cast("Trainer", trainer), schedule


def test_captured() -> None:
    """The epoch, the score and the scheduler's position all come back off the live objects."""
    trainer, schedule = _trainer()
    schedule.step()
    state = ResumeState.captured(trainer, np.random.default_rng(SEED), epoch=3, best=0.86)

    assert state.epoch == 3
    assert state.best == 0.86
    assert state.schedule is not None
    assert state.schedule["last_epoch"] == 1


def test_beside(tmp_path: Path) -> None:
    """The trajectory is named from the checkpoint, so the two cannot be given different stems."""
    assert ResumeState.beside(tmp_path / "member.pt") == tmp_path / ("member" + Artefact.RESUME)


def test_write(tmp_path: Path) -> None:
    """Writing puts the file beside the checkpoint, under the checkpoint's own stem."""
    destination = tmp_path / "member.pt"
    ResumeState.captured(_trainer()[0], np.random.default_rng(SEED), epoch=2, best=0.8).write(destination)

    assert ResumeState.beside(destination).exists()


def test_read(tmp_path: Path) -> None:
    """What comes back off disk is what was written, and a missing trajectory answers None."""
    destination = tmp_path / "member.pt"
    assert ResumeState.read(destination) is None

    ResumeState.captured(_trainer()[0], np.random.default_rng(SEED), epoch=5, best=0.9).write(destination)
    held = ResumeState.read(destination)

    assert held is not None
    assert (held.epoch, held.best) == (5, 0.9)


def test_restore() -> None:
    """A stepped-on schedule and a drawn-from generator both return to where they were captured."""
    (trainer, schedule), generator = _trainer(), np.random.default_rng(SEED)
    state = ResumeState.captured(trainer, generator, epoch=0, best=0.5)
    expected = generator.random(3)

    for _ in range(4):
        schedule.step()
    generator.random(3)
    state.restore(trainer, generator)

    assert schedule.last_epoch == 0
    assert np.array_equal(generator.random(3), expected)
