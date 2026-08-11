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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor, nn

from kneemri.model import StudyModel
from kneemri.study_cache import CachedCorpus
from kneemri.targets import Targets

HEAD_LR = 1e-3
BACKBONE_LR = 8e-6
WEIGHT_DECAY = 0.02
UNFREEZE_LAST = 6
SEED = 2026

ONE_CYCLE_WARMUP = 0.15
"""Fraction of the run spent warming up before the decay — protects the pretrained encoder early."""

GROUP = 3
"""Consecutive slices stacked as the encoder's three channels."""


@dataclass(frozen=True)
class AugmentConfig:
    """How hard to perturb a training image, and which perturbations to apply at all.

    The geometric widths are the public baseline's. The INTENSITY block is not: it comes from the
    sibling cardiac-MRI project, where gamma / contrast / blur / noise were the transforms that
    measurably closed a cross-vendor gap (`../cardiac-seg/core/data/dynamic/augment.py`). This corpus
    has the same problem in a harder form — reports in nine languages means many countries and many
    scanners — and the public baseline augments geometry only.

    NO FLIPS, ever. A horizontal flip undoes the laterality normalisation that five of the twelve
    targets depend on. The sibling project flips both axes freely because a cardiac label is not
    side-defined; here it would teach the model that medial and lateral are the same thing.

    One inherited finding does NOT transfer, and the reason is worth keeping: cardiac-seg measured
    translation as NEUTRAL and turned it off, because a fully-convolutional U-Net is
    translation-equivariant so absolute position barely matters. Our encoder is a ViT with position
    embeddings and is NOT equivariant, so translation is kept on here.
    """

    rotation_degrees: float = 8.0
    scale_jitter: float = 0.08
    shift_jitter: float = 0.05
    gain_jitter: float = 0.10
    gamma: tuple[float, float] = (0.7, 1.5)
    gamma_probability: float = 0.3
    blur_probability: float = 0.2
    contrast: tuple[float, float] = (0.85, 1.15)
    noise: float = 0.02
    """Noise is far below the sibling's 0.08: theirs acts on z-scored images, ours on [0, 1]."""

    @classmethod
    def geometric_only(cls) -> "AugmentConfig":
        """The public baseline's augmentation: rigid jitter and one global gain, no intensity ops.

        This is the DEFAULT, so that enabling the intensity block is an explicit, ablatable decision
        rather than a silent change of recipe. It also keeps a run reproducible against members that
        were fitted before the block existed.
        """
        return cls(gamma_probability=0.0, blur_probability=0.0, contrast=(1.0, 1.0), noise=0.0)


ROTATION_DEGREES = 8.0
SCALE_JITTER = 0.08
SHIFT_JITTER = 0.05
INTENSITY_JITTER = 0.10


class Trainer:
    """Fits a `StudyModel` on a cached corpus and scores it on a held-out fold."""

    _BLUR = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
    """A 3x3 Gaussian, normalised so a blur cannot change the image's overall level."""

    def __init__(
        self,
        model: StudyModel,
        device: torch.device,
        unfreeze_last: int = UNFREEZE_LAST,
        augment_config: AugmentConfig | None = None,
        *,
        normalise_by_count: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.unfreeze_last = unfreeze_last
        self.augment_config = augment_config or AugmentConfig.geometric_only()
        self.normalise_by_count = normalise_by_count
        groups = self.model.parameter_groups(HEAD_LR, BACKBONE_LR, unfreeze_last)
        self.optimiser = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        self.schedule: torch.optim.lr_scheduler.OneCycleLR | None = None
        self.torch_generator = torch.Generator(device=device).manual_seed(SEED)

    def one_cycle(self, total_steps: int) -> None:
        """Attach a one-cycle schedule over the whole run, warming up then decaying to near zero.

        MEASURED REASON: at a constant learning rate our five folds peaked at epochs 13/13/10/12/11 and
        gained ~0 over their final five — saturated at ~12 epochs. A run that improves out to 60 epochs
        is not training longer, it is training on a decaying learning rate. `pct_start=0.15` spends the
        first sixth warming up, which also protects the pretrained encoder from the first few hundred
        steps at full rate.
        """
        self.schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.optimiser,
            max_lr=[HEAD_LR, BACKBONE_LR],
            total_steps=max(total_steps, 1),
            pct_start=ONE_CYCLE_WARMUP,
        )

    def augment(self, images: Float[Tensor, "n c h w"], generator: torch.Generator) -> Float[Tensor, "n c h w"]:
        """Rigid jitter, then intensity — never a flip, which laterality normalisation forbids.

        Every transform is per-sample and vectorised: parameters come from one `torch.rand(n, ...)` and
        probabilistic transforms are applied with a mask (`out = do*transformed + (1-do)*original`)
        rather than a Python branch, so a batch costs one kernel per transform and not one per image.
        """
        warped = self._warp(images, generator)
        return self._intensity(warped, generator)

    def _warp(self, images: Float[Tensor, "n c h w"], generator: torch.Generator) -> Float[Tensor, "n c h w"]:
        """Rotation, scale and translation as a single affine resample."""
        config, count, device = self.augment_config, images.shape[0], images.device
        angles = (torch.rand(count, generator=generator, device=device) * 2 - 1) * np.deg2rad(config.rotation_degrees)
        scales = 1 + (torch.rand(count, generator=generator, device=device) * 2 - 1) * config.scale_jitter
        shifts = (torch.rand(count, 2, generator=generator, device=device) * 2 - 1) * config.shift_jitter
        cosine, sine = torch.cos(angles) / scales, torch.sin(angles) / scales
        theta = torch.zeros(count, 2, 3, device=device)
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cosine, -sine, shifts[:, 0]
        theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sine, cosine, shifts[:, 1]
        grid = nn.functional.affine_grid(theta, list(images.shape), align_corners=False)
        # `border` rather than `zeros`: a rotated knee should run out into tissue-like values, not into a
        # black wedge the encoder can use to infer how far the image was rotated.
        return nn.functional.grid_sample(images, grid, align_corners=False, padding_mode="border")

    def _intensity(self, images: Float[Tensor, "n c h w"], generator: torch.Generator) -> Float[Tensor, "n c h w"]:
        """Gain, gamma, blur, contrast and noise — the scanner-variation axis, on [0, 1] inputs."""
        config, count, device = self.augment_config, images.shape[0], images.device
        shape = (count, 1, 1, 1)

        gain = 1 + (torch.rand(shape, generator=generator, device=device) * 2 - 1) * config.gain_jitter
        images = images * gain

        low, high = config.gamma
        exponent = torch.rand(shape, generator=generator, device=device) * (high - low) + low
        apply_gamma = (torch.rand(shape, generator=generator, device=device) < config.gamma_probability).float()
        images = apply_gamma * images.clamp(0, 1) ** exponent + (1 - apply_gamma) * images

        apply_blur = (torch.rand(shape, generator=generator, device=device) < config.blur_probability).float()
        channels = images.shape[1]
        kernel = self._BLUR.to(device, images.dtype).view(1, 1, 3, 3).expand(channels, 1, 3, 3)
        blurred = nn.functional.conv2d(images, kernel, padding=1, groups=channels)
        images = apply_blur * blurred + (1 - apply_blur) * images

        low, high = config.contrast
        contrast = torch.rand(shape, generator=generator, device=device) * (high - low) + low
        # Contrast scales about the image's own mean, so it changes the SPREAD of intensities without
        # also shifting the level — which `gain` above is already responsible for.
        mean = images.mean(dim=(1, 2, 3), keepdim=True)
        images = (images - mean) * contrast + mean

        if config.noise > 0:
            images = images + torch.randn(images.shape, generator=generator, device=device) * config.noise
        return images.clamp(0, 1)

    def augment_bag(self, images: UInt8[Tensor, "b n g h w"]) -> UInt8[Tensor, "b n g h w"]:
        """Augment a batch of slot bags, staying in uint8 so the model's own forward is untouched.

        The warp needs floats, but `StudyModel.forward` owns the byte scaling and the ImageNet
        normalisation — and its fingerprint is the identity check that stops a member being run under a
        reading it was not fitted under. So the augmentation converts, warps and converts back rather
        than handing the model a tensor in a different space. The requantisation is lossless in the
        sense that matters: the cache is already uint8.

        Each slot image is warped INDEPENDENTLY (the bag is folded into the batch axis). That is
        deliberate — the six slots are separate acquisitions of the same knee, not one volume, so a
        shared jitter would imply a registration between them that does not exist.
        """
        batch, slots = images.shape[:2]
        flat = images.reshape(batch * slots, *images.shape[2:]).float().div(255.0)
        warped = self.augment(flat, self.torch_generator)
        return (warped * 255).round().clamp(0, 255).to(torch.uint8).reshape(images.shape)

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
        # `images[batch, :, s:s+G]` in ONE indexing op, never `images[batch][:, :, s:s+G]`. The latter
        # fancy-indexes first, materialising all twelve cached slices (97.5 MB per batch) before
        # discarding three quarters of them: measured 35 ms against 3.3 ms, or about ten seconds of
        # every forty-second epoch spent reading data that is then thrown away.
        self.model.train()
        total, steps = 0.0, 0
        for batch in self.batches(positions, size, generator, shuffle=True):
            start = int(generator.integers(0, corpus.slices - GROUP + 1))
            pixels = torch.from_numpy(np.ascontiguousarray(corpus.images[batch, :, start : start + GROUP])).to(
                self.device
            )
            pixels = self.augment_bag(pixels)
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
            if self.schedule is not None:
                self.schedule.step()
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
                pixels = torch.from_numpy(np.ascontiguousarray(corpus.images[batch, :, middle : middle + GROUP])).to(
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
        """Confidence-weighted BCE, normalised either by total weight or by element count.

        BY WEIGHT (ours): a batch carrying little confidence does not take a full-sized step on almost
        no evidence. BY COUNT (the public baseline's `.mean()`): the step size is a fixed scale that
        does not depend on how confidently the batch happened to be labelled.

        The two are not interchangeable in magnitude. With most weights below one, dividing by their
        sum makes our gradients roughly twice theirs for the same batch — which interacts with the
        learning-rate schedule, so the pair has to be measured together rather than one at a time.
        """
        elementwise = nn.functional.binary_cross_entropy_with_logits(logits, scores, reduction="none")
        weighted = elementwise * weights
        if self.normalise_by_count:
            return weighted.mean()
        return weighted.sum() / weights.sum().clamp(min=1.0)

    def save_partial(self, destination: Path, state: dict[str, Tensor], holdout: float, epoch: int) -> None:
        """Write the best weights so far, atomically, the moment they improve.

        Holding the only copy in memory until the final epoch is how a run that has already produced
        its best epoch loses everything — which the OneCycle off-by-one demonstrated by raising on the
        very last step of a thirty-epoch run. Temp-then-rename means a crash mid-write cannot leave a
        truncated file behind either.
        """
        partial = destination.with_suffix(".partial")
        temporary = partial.with_suffix(".tmp")
        torch.save({"model": state, "holdout": holdout, "epoch": epoch}, temporary)
        temporary.replace(partial)
