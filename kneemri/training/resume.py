"""Continue a killed run's OWN schedule, rather than starting a new one from its weights.

`WarmStart` is not this and cannot be made into it: it loads weights and builds a FRESH `OneCycleLR`,
so a run re-entered through it follows a different learning-rate trajectory than the one it was queued
under. An arm resumed that way carries two knobs — the one it was armed to test and the restart — and
is no longer comparable to the control it was priced against.

WHAT IS RESTORED, AND WHY EACH ONE IS LOAD-BEARING. Weights alone reproduce the model and none of the
trajectory. `AdamW`'s moments decide the next step's SIZE; `OneCycleLR`'s step count decides its RATE;
`GradScaler`'s scale decides whether that step is taken at all under mixed precision; and the data
order and every dropout mask are drawn from the generators, so a run resumed with fresh streams sees a
different corpus order through a different network. Any one of them missing turns a resume into a warm
start wearing a resume's name.

THE RESUME POINT IS THE LAST KEPT EPOCH, never the last epoch attempted. `Fitting.keep` writes the
weights and this state together, so the two can never describe different epochs. The cost is that
epochs run after the last improvement are re-run; that is bounded by the patience window and is the
price of the guarantee.

THIS STATE IS NOT PART OF A MEMBER. It lives in its own file beside the checkpoint because a member is
weights plus identity plus a reading, and that is what `MemberPackage` consumes and what ships. The
optimiser moments are roughly twice the size of the weights and mean nothing to inference.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from kneemri.runs.artefact import Artefact
from kneemri.training.durable import Durable
from kneemri.training.trainer import Trainer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResumeState:
    """Everything a run needs to take its NEXT step exactly as it would have taken it uninterrupted."""

    epoch: int
    """The index of the epoch this state was captured after; a resumed run starts at `epoch + 1`."""

    best: float
    """The score the kept epoch was ranked by, so a resumed run cannot keep a worse later epoch."""

    optimiser: dict[str, object]
    schedule: dict[str, object] | None
    scaler: dict[str, object]

    order: dict[str, object]
    """The numpy bit generator's state — the data order, which is drawn from it every epoch."""

    torch_rng: torch.Tensor
    cuda_rng: list[torch.Tensor]

    trainer_rng: torch.Tensor
    """The trainer's OWN generator: augmentation, patch erasing and slot dropout all draw from it."""

    @classmethod
    def captured(cls, trainer: Trainer, generator: np.random.Generator, epoch: int, best: float) -> "ResumeState":
        """Take the trajectory as it stands, alongside the weights the caller is about to write."""
        return cls(
            epoch=epoch,
            best=best,
            optimiser=trainer.optimiser.state_dict(),
            schedule=None if trainer.schedule is None else trainer.schedule.state_dict(),
            scaler=trainer.scaler.state_dict(),
            order=dict(generator.bit_generator.state),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            trainer_rng=trainer.torch_generator.get_state(),
        )

    @staticmethod
    def beside(destination: Path) -> Path:
        """Where a checkpoint's trajectory file sits — named from the checkpoint, never from the arm."""
        return Artefact.RESUME.beside(destination)

    def write(self, destination: Path) -> None:
        """Put this trajectory on the platter beside the weights it belongs to."""
        Durable.save(
            {
                "epoch": self.epoch,
                "best": self.best,
                "optimiser": self.optimiser,
                "schedule": self.schedule,
                "scaler": self.scaler,
                "order": self.order,
                "torch_rng": self.torch_rng,
                "cuda_rng": self.cuda_rng,
                "trainer_rng": self.trainer_rng,
            },
            self.beside(destination),
        )

    @classmethod
    def read(cls, destination: Path) -> "ResumeState | None":
        """The trajectory beside `destination`, or None when there is nothing to continue.

        A missing file is the ordinary case — every run that has not been interrupted — so it answers
        None rather than raising. An UNREADABLE one raises: a trajectory that cannot be loaded must not
        be silently downgraded into a cold start, which is the failure this module exists for.
        """
        path = cls.beside(destination)
        if not path.exists():
            return None
        held = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            epoch=int(held["epoch"]),
            best=float(held["best"]),
            optimiser=held["optimiser"],
            schedule=held["schedule"],
            scaler=held["scaler"],
            order=held["order"],
            torch_rng=held["torch_rng"],
            cuda_rng=held["cuda_rng"],
            trainer_rng=held["trainer_rng"],
        )

    def restore(self, trainer: Trainer, generator: np.random.Generator) -> None:
        """Put the trajectory back, so the next step is the one the killed run would have taken.

        The scheduler must already have been BUILT: `OneCycleLR` is constructed from a total step count
        and a `pct_start`, and loading a state dict into a scheduler shaped for a different budget would
        continue a curve this run never had.
        """
        trainer.optimiser.load_state_dict(self.optimiser)
        if self.schedule is not None and trainer.schedule is not None:
            trainer.schedule.load_state_dict(self.schedule)
        trainer.scaler.load_state_dict(self.scaler)
        generator.bit_generator.state = self.order
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_rng)
        trainer.torch_generator.set_state(self.trainer_rng)
        logger.info("resumed after epoch %d (best %.4f) — continuing the same schedule", self.epoch, self.best)
