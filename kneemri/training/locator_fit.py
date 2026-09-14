"""Fit the tissue `Locator` on SKM-TEA's sagittal DESS slices, and score it on SKM-TEA's own test split.

`python -m kneemri.training.locator_fit --epochs 40 --out locator_sag.pt` reads what
`core.data.corpus.skmtea_slices` wrote and saves the state dict beside it. SKM-TEA's train / val / test
split is the one its v1.0.0 annotation files declare; nothing here draws a split of its own.

THE TRAINING DATA IS ONE SEQUENCE AND THE TARGET IS FOUR. SKM-TEA is DESS only; the RSNA sagittal slots
are fat-suppressed fluid, fluid without suppression, and T1. Measured on 30 RSNA SAG_T1 tiles with no
intensity remapping: about 13 came back with no tissue drawn at all. `LocatorAugment.intensity` pushes a
random nine-knot lookup table through each slice — monotone or not — so contrast ORDER stops being a
cue the head can lean on; after it, 6 of 150 SAG_T1 tiles were blank (`interpretations/kneemri/
2026-09-14_roi_locator_audit.md`).

The geometry draws a crop of 0.4-1.0 of the side anywhere in the frame, plus a left-right flip, because
the RSNA field of view is tighter than SKM-TEA's 160 mm and a knee may be either side.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, UInt8
from torch import Tensor

from core import cli
from core.data.corpus.skmtea_slices import LOCATOR
from kneemri.models.backbone import ENCODERS, SMALL
from kneemri.models.locator import BYTE, CLASSES, EVAL_BATCH, Locator, SliceCorpus

logger = logging.getLogger(__name__)

BATCH = 16
TRUNK_RATE = 2e-5
HEAD_RATE = 1e-3
WEIGHT_DECAY = 0.01
CLASS_WEIGHTS = (0.2, 1.0, 1.0, 1.0, 1.5)
"""Background is most of every slice; the meniscus is the smallest tissue and the one `.5` needs."""
SMOOTH = 1.0
SCALE = (0.4, 1.0)
GAMMA = 0.9
"""log-gamma drawn in +-0.9: a gamma of 0.4 to 2.5."""
KNOTS = 9
REMAP = 0.6
MONOTONE = 0.5
GAIN = (0.7, 1.3)
NOISE = 0.05
HALF = 0.5
EPOCHS = 40


@dataclass(frozen=True)
class LocatorAugment:
    """Random geometry and intensity, drawn per slice from one generator."""

    generator: torch.Generator

    def uniform(self, low: float, high: float, *shape: int, device: torch.device) -> Float[Tensor, "*shape"]:
        return torch.rand(*shape, generator=self.generator, device=device) * (high - low) + low

    def geometry(
        self, slices: Float[Tensor, "n h w"], labels: Int[Tensor, "n h w"]
    ) -> tuple[Float[Tensor, "n h w"], Int[Tensor, "n h w"]]:
        count, device = slices.shape[0], slices.device
        scale = self.uniform(*SCALE, count, device=device)
        flip = torch.where(self.uniform(0, 1, count, device=device) < HALF, -1.0, 1.0)
        theta = torch.zeros(count, 2, 3, device=device)
        theta[:, 0, 0] = scale * flip
        theta[:, 1, 1] = scale
        theta[:, :, 2] = self.uniform(-1, 1, count, 2, device=device) * (1 - scale)[:, None]
        grid = F.affine_grid(theta, [count, 1, *slices.shape[1:]], align_corners=False)
        moved = F.grid_sample(slices[:, None], grid, align_corners=False)[:, 0]
        drawn = F.grid_sample(labels[:, None].float(), grid, mode="nearest", align_corners=False)[:, 0]
        return moved, drawn.long()

    def intensity(self, slices: Float[Tensor, "n h w"]) -> Float[Tensor, "n h w"]:
        """0-1 slices through a gamma, a random lookup table, a gain and noise; 0-1 slices back."""
        count, device = slices.shape[0], slices.device
        unit = slices.clamp(0, 1) ** torch.exp(self.uniform(-GAMMA, GAMMA, count, 1, 1, device=device))
        knots = self.uniform(0, 1, count, KNOTS, device=device)
        knots = torch.where(self.uniform(0, 1, count, 1, device=device) < MONOTONE, knots.sort(1).values, knots)
        position = unit * (KNOTS - 1)
        below = position.floor().long().clamp(max=KNOTS - 2)
        fraction = position - below
        rows = torch.arange(count, device=device).view(count, 1, 1).expand_as(below)
        looked = knots[rows, below] * (1 - fraction) + knots[rows, below + 1] * fraction
        unit = torch.where(self.uniform(0, 1, count, 1, 1, device=device) < REMAP, looked, unit)
        noise = torch.randn(unit.shape, generator=self.generator, device=device)
        gain = self.uniform(*GAIN, count, 1, 1, device=device)
        unit = unit * gain + noise * self.uniform(0, NOISE, count, 1, 1, device=device)
        return unit.clamp(0, 1)

    def __call__(
        self, slices: UInt8[Tensor, "n h w"], labels: Int[Tensor, "n h w"]
    ) -> tuple[Float[Tensor, "n h w"], Int[Tensor, "n h w"]]:
        moved, drawn = self.geometry(slices.float() / BYTE, labels)
        return self.intensity(moved) * BYTE, drawn


@dataclass
class LocatorFit:
    """One locator, one corpus, one generator: the fit and the Dice it is read by."""

    locator: Locator
    corpus: SliceCorpus
    device: torch.device
    generator: torch.Generator

    @staticmethod
    def loss(logits: Float[Tensor, "n classes h w"], labels: Int[Tensor, "n h w"]) -> Float[Tensor, ""]:
        """Weighted cross-entropy plus a soft Dice over the four tissues, which the background cannot swamp."""
        weights = torch.tensor(CLASS_WEIGHTS, device=logits.device)
        entropy = F.cross_entropy(logits.float(), labels, weight=weights)
        probability = logits.float().softmax(1)[:, 1:]
        truth = F.one_hot(labels, len(CLASSES)).permute(0, 3, 1, 2)[:, 1:].float()
        overlap = (probability * truth).sum((0, 2, 3))
        dice = (2 * overlap + SMOOTH) / (probability.sum((0, 2, 3)) + truth.sum((0, 2, 3)) + SMOOTH)
        return entropy + 1 - dice.mean()

    def dice(self, split: str) -> dict[str, float]:
        """Per-tissue Dice over every slice of `split`, first echo, pooled over slices."""
        chosen = self.corpus.indices(split)
        overlap = torch.zeros(len(CLASSES))
        total = torch.zeros(len(CLASSES))
        self.locator.eval()
        for start in range(0, len(chosen), EVAL_BATCH):
            part = chosen[start : start + EVAL_BATCH]
            slices, labels = self.corpus.batch(part, np.zeros_like(part), self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                drawn = self.locator.predict(slices)
            for tissue in range(1, len(CLASSES)):
                overlap[tissue] += ((drawn == tissue) & (labels == tissue)).sum().item()
                total[tissue] += ((drawn == tissue).sum() + (labels == tissue).sum()).item()
        return {CLASSES[t]: round(2 * overlap[t].item() / max(total[t].item(), 1), 3) for t in range(1, len(CLASSES))}

    def fit(self, epochs: int, batch: int = BATCH) -> float:
        augment = LocatorAugment(torch.Generator(self.device).manual_seed(0))
        train = self.corpus.indices("train")
        steps = max(epochs * (len(train) // batch), 1)
        optimizer = torch.optim.AdamW(self.locator.groups(TRUNK_RATE, HEAD_RATE), weight_decay=WEIGHT_DECAY)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[TRUNK_RATE, HEAD_RATE], total_steps=steps)
        loss = torch.zeros(())
        for epoch in range(epochs):
            self.locator.train()
            order = train[torch.randperm(len(train), generator=self.generator).numpy()]
            for start in range(0, len(order) - batch + 1, batch):
                part = order[start : start + batch]
                echoes = torch.randint(0, self.corpus.images.shape[1], (len(part),), generator=self.generator).numpy()
                slices, labels = augment(*self.corpus.batch(part, echoes, self.device))
                with torch.autocast(self.device.type, dtype=torch.bfloat16):
                    loss = self.loss(self.locator(slices), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                schedule.step()
            logger.info("epoch %d loss %.3f val %s", epoch, loss.item(), self.dice("val"))
        return float(loss.item())


def main() -> None:
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--epochs", type=int, default=EPOCHS)
    command.parser.add_argument("--out", required=True, help="checkpoint name, written beside the slices")
    arguments = command.parsed()
    directory = command.rooted(arguments).processed() / LOCATOR
    device = torch.device("cuda")
    fit = LocatorFit(
        Locator.pretrained(ENCODERS[SMALL]).to(device),
        SliceCorpus.read(directory),
        device,
        torch.Generator().manual_seed(0),
    )
    fit.fit(arguments.epochs)
    command.report([f"TEST {fit.dice('test')}"])
    torch.save(fit.locator.state_dict(), directory / arguments.out)
    logger.info("LOCATOR FIT COMPLETE")


if __name__ == "__main__":
    main()
