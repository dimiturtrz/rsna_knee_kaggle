"""Adapting the encoder and fitting the head, on report-derived targets weighted by confidence.

Two restraints on the encoder, both from the shape of the supervision rather than from taste:

**Only the last blocks move.** Early blocks of a vision transformer are generic edge and texture
filters; late blocks carry semantics. With 58 image-read labels and the rest derived from text, there
may not be enough supervision to improve the early ones and there is certainly enough to damage them.

**The encoder learns far more slowly than the head.** The head is random at initialisation and has
everything to learn; the encoder starts from a good solution and needs only to be moved off it. A single
learning rate either leaves the head untrained or destroys the encoder in the first few hundred steps.

Augmentation is rigid jitter plus intensity only — **no flips**. A horizontal flip would undo the
laterality normalisation that five of the twelve targets depend on, turning a medial meniscus into a
lateral one and teaching the model that the distinction is noise.
"""

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor, nn

from kneemri.model import TARGETS, StudyModel
from kneemri.study_cache import CachedCorpus
from kneemri.targets import Targets

HEAD_LR = 1e-3
BACKBONE_LR = 8e-6
WEIGHT_DECAY = 0.02
UNFREEZE_LAST = 6

GROUP = 3
"""Consecutive slices stacked as the encoder's three channels."""

MIDPOINT = 0.5
"""Where a graded target is cut so a holdout AUC exists at all — never a submitted threshold."""

ROTATION_DEGREES = 8.0
SCALE_JITTER = 0.08
SHIFT_JITTER = 0.05
INTENSITY_JITTER = 0.10


class Trainer:
    """Fits a `StudyModel` on a cached corpus and scores it on a held-out fold."""

    def __init__(self, model: StudyModel, device: torch.device, unfreeze_last: int = UNFREEZE_LAST) -> None:
        self.model = model.to(device)
        self.device = device
        self.unfreeze_last = unfreeze_last
        groups = self.model.parameter_groups(HEAD_LR, BACKBONE_LR, unfreeze_last)
        self.optimiser = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def augment(self, images: Float[Tensor, "n c h w"], generator: torch.Generator) -> Float[Tensor, "n c h w"]:
        """Rigid jitter and a brightness scale — never a flip, which laterality normalisation forbids."""
        count = images.shape[0]
        angles = (torch.rand(count, generator=generator, device=images.device) * 2 - 1) * np.deg2rad(ROTATION_DEGREES)
        scales = 1 + (torch.rand(count, generator=generator, device=images.device) * 2 - 1) * SCALE_JITTER
        shifts = (torch.rand(count, 2, generator=generator, device=images.device) * 2 - 1) * SHIFT_JITTER
        cosine, sine = torch.cos(angles) / scales, torch.sin(angles) / scales
        theta = torch.zeros(count, 2, 3, device=images.device)
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cosine, -sine, shifts[:, 0]
        theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sine, cosine, shifts[:, 1]
        grid = nn.functional.affine_grid(theta, list(images.shape), align_corners=False)
        warped = nn.functional.grid_sample(images, grid, align_corners=False, padding_mode="border")
        gain = 1 + (torch.rand(count, 1, 1, 1, generator=generator, device=images.device) * 2 - 1) * INTENSITY_JITTER
        return (warped * gain).clamp(0, 1)

    def batches(
        self, positions: NDArray[np.int64], size: int, generator: np.random.Generator, *, shuffle: bool
    ) -> list[NDArray[np.int64]]:
        """Positions split into batches: shuffled for training, left in order for evaluation."""
        order = generator.permutation(positions) if shuffle else positions
        return [order[start : start + size] for start in range(0, len(order), size)]

    def fit_epoch(
        self,
        corpus: CachedCorpus,
        targets: Targets,
        positions: NDArray[np.int64],
        size: int,
        generator: np.random.Generator,
    ) -> float:
        """One pass over the training positions, returning the mean weighted loss.

        The window start is drawn per BATCH rather than per study: every study in a step is read at the
        same depth, which keeps the batch a single contiguous slice of the memory map instead of a
        gather, and the randomness across steps still covers the stack.
        """
        self.model.train()
        total, steps = 0.0, 0
        for batch in self.batches(positions, size, generator, shuffle=True):
            start = int(generator.integers(0, corpus.slices - GROUP + 1))
            pixels = torch.from_numpy(np.ascontiguousarray(corpus.images[batch][:, :, start : start + GROUP])).to(
                self.device
            )
            flags = torch.from_numpy(corpus.mask[batch]).to(self.device)
            score, weight = targets.rows(batch)
            with torch.autocast("cuda", enabled=self.device.type == "cuda"):
                logits = self.model(pixels, flags, corpus.image_size)
                loss = self.loss(
                    logits.float(),
                    torch.from_numpy(score).to(self.device),
                    torch.from_numpy(weight).to(self.device),
                )
            self.optimiser.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimiser)
            self.scaler.update()
            total += float(loss.detach())
            steps += 1
        return total / max(steps, 1)

    def predict(
        self,
        corpus: CachedCorpus,
        positions: NDArray[np.int64],
        size: int,
        generator: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Probabilities for a set of positions, read from the middle window of the stack."""
        self.model.eval()
        middle = (corpus.slices - GROUP) // 2
        chunks = []
        with torch.no_grad(), torch.autocast("cuda", enabled=self.device.type == "cuda"):
            for batch in self.batches(positions, size, generator, shuffle=False):
                pixels = torch.from_numpy(np.ascontiguousarray(corpus.images[batch][:, :, middle : middle + GROUP])).to(
                    self.device
                )
                flags = torch.from_numpy(corpus.mask[batch]).to(self.device)
                chunks.append(torch.sigmoid(self.model(pixels, flags, corpus.image_size).float()).cpu().numpy())
        return np.concatenate(chunks).astype(np.float64)

    def loss(
        self,
        logits: Float[Tensor, "b o"],
        scores: Float[Tensor, "b o"],
        weights: Float[Tensor, "b o"],
    ) -> Float[Tensor, ""]:
        """Confidence-weighted BCE: a target the report never mentioned pulls with weight zero.

        Normalised by the total weight rather than by the element count, so a batch that happens to
        carry little confidence does not take a full-sized step on almost no evidence.
        """
        elementwise = nn.functional.binary_cross_entropy_with_logits(logits, scores, reduction="none")
        return (elementwise * weights).sum() / weights.sum().clamp(min=1.0)

    def area_under_curve(self, scores: NDArray[np.float64], truth: NDArray[np.int64]) -> float:
        """ROC AUC by rank, which is its definition — no sklearn, no calibration, no thresholds."""
        positives, negatives = int(truth.sum()), int((1 - truth).sum())
        if positives == 0 or negatives == 0:
            return float("nan")
        order = scores.argsort().argsort() + 1
        return float((order[truth == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))

    def evaluate(self, predictions: NDArray[np.float64], scores: NDArray[np.float32]) -> dict[str, float]:
        """Per-label AUC on the holdout, against the graded targets binarised at their midpoint.

        Binarising is only so an AUC exists at all; it decides which epoch is kept and never a submitted
        score, and the metric is invariant to any increasing map of the predictions anyway.
        """
        return {
            target: self.area_under_curve(predictions[:, column], (scores[:, column] > MIDPOINT).astype(np.int64))
            for column, target in enumerate(TARGETS)
        }
