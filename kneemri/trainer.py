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

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor, nn

from kneemri.ema import DECAY, LiveWeights, WeightAverage
from kneemri.heartbeat import Heartbeat
from kneemri.model import StudyModel
from kneemri.objective import Objective
from kneemri.report_supervision import NoReportSupervision, ReportSupervision
from kneemri.study_cache import CachedCorpus
from kneemri.targets import Targets

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024**3
"""So a VRAM budget can be stated in the unit the card is sold in."""

WEIGHT_DECAY = 0.02
SEED = 2026


@dataclass(frozen=True)
class DeviceBudget:
    """The card a run may use, and how much of it it is allowed to hold.

    A cap looks like a restriction and is actually a throughput fix, which is why it is a first-class
    object rather than three lines in `main`. MEASURED 2026-08-11 on a 32 GB 5090, whole-stack reading
    at batch 12:

        uncapped   32.0 GB held, 145 W of 575, 100% "utilisation", NO epoch in fifteen minutes
        18 GB cap  19.2 GB held, 470-514 W,     99% utilisation,    training normally

    Windows pages GPU memory over PCIe rather than raising OOM, so a run that does not fit does not
    fail - it crawls, while `nvidia-smi` reports full utilisation because a kernel is resident. Power
    draw is the honest signal: a card doing real work draws its budget. Holding the run to a share of
    the card keeps it inside physical VRAM, and turns the worst failure mode we have into an immediate
    OutOfMemoryError.

    The second reason is coexistence: another project trains on this same card, and an uncapped run
    takes all of it.
    """

    vram_gb: float = 0.0
    """0 leaves the card uncapped."""

    def select(self) -> torch.device:
        """The device to train on, already held to this budget."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.vram_gb and device.type == "cuda":
            # An INDEX, not torch.device("cuda"): both torch calls reject an indexless device.
            card = torch.cuda.current_device()
            whole = torch.cuda.get_device_properties(card).total_memory / BYTES_PER_GB
            torch.cuda.set_per_process_memory_fraction(min(self.vram_gb / whole, 1.0), card)
            logger.info("VRAM capped at %.1f GB of %.1f GB", self.vram_gb, whole)
        return device


@dataclass(frozen=True)
class Adaptation:
    """How much of the encoder moves, and how fast — one object because they cannot be read apart.

    The head is random at initialisation and has everything to learn; the encoder starts from a good
    solution and needs only to be moved off it. That argument sets the 125:1 default ratio, and it was
    inherited from the reference rather than measured.

    Depth and rate are ENTANGLED, which is why they stopped being three constants. Our bracket found
    unfreezing every block worth +0.0008, and a null result at a backbone rate of 8e-6 cannot
    distinguish "more trainable blocks do not help" from "no block was going to move anyway". Asking
    how many blocks should train is only a real question at a rate where a block moves.
    """

    head_lr: float = 1e-3
    backbone_lr: float = 8e-6
    unfreeze_last: int = 6
    """Six of twelve blocks — inherited from the reference, and only meaningful beside the rate above."""


ONE_CYCLE_WARMUP = 0.15
"""Fraction of the run spent warming up before the decay — protects the pretrained encoder early."""

GROUP = 3
"""Consecutive slices stacked as the encoder's three channels."""

WHOLE_STACK = 0
"""`window=0` means read every cached slice at once — the whole stack rather than one window of it."""


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
        adaptation: Adaptation | None = None,
        augment_config: AugmentConfig | None = None,
        objective: Objective | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.adaptation = adaptation or Adaptation()
        self.augment_config = augment_config or AugmentConfig.geometric_only()
        self.objective = objective or Objective()
        groups = self.model.parameter_groups(
            self.adaptation.head_lr, self.adaptation.backbone_lr, self.adaptation.unfreeze_last
        )
        self.optimiser = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        self.schedule: torch.optim.lr_scheduler.OneCycleLR | None = None
        self.torch_generator = torch.Generator(device=device).manual_seed(SEED)
        # How many consecutive slices one read covers. Three is the RGB trick; anything else needs
        # per-slice pooling, because channel stacking has exactly three channels to stack into.
        self.window = GROUP
        # Attached rather than passed, for the same reason the schedule is: both are policies a run may
        # or may not want, and a sixth constructor argument trips the argument-count gate. The default
        # is a no-op object rather than None, so no call site has to test for one.
        self.average: LiveWeights = LiveWeights()
        self.heartbeat = Heartbeat()
        self.reports: NoReportSupervision = NoReportSupervision()
        # None means "read the cache at its own resolution", which is what every run did before
        # resampling existed. See `resample_to`.
        self.read_size: int | None = None

    def report_supervision(self, reports: ReportSupervision) -> None:
        """Also supervise the study representation by its report embedding."""
        self.reports = reports

    def whole_stack(self, slices: int) -> None:
        """Read the ENTIRE cached stack in one step instead of one sampled window of it.

        Only meaningful with per-slice pooling, which encodes each slice alone and takes the
        per-dimension maximum — so a slot's embedding stops depending on which window was drawn, and a
        finding is no longer invisible unless it happens to fall inside the sampled three.

        The evidence for trying it is the strongest single number in this project: averaging ten
        overlapping windows at INFERENCE is worth +0.0156, far more than any method change measured
        here. Training already had that information available and was throwing it away each step.

        Costs one encoder pass per slice rather than per group — about 4x a step at twelve slices.
        """
        if not self.model.per_slice:
            message = f"whole-stack reading needs per-slice pooling, not {self.model.slice_pooling!r}"
            raise ValueError(message)
        self.window = slices

    def resample_to(self, size: int) -> None:
        """Train at `size` px from whatever resolution the cache holds, instead of the cache's own.

        Decoupling the two makes resolution a FREE variable: the encoder already interpolates in
        `slot_features`, so a 448 px cache can be read at 392 without a second 45-minute decode, and
        every size sees the same pixels through a different sampling grid rather than a different crop.

        It exists because resolution had a hard ceiling otherwise. Whole-stack at 448 px needs ~34 GB
        on a 32 GB card — tokens go as the square of the side — and every way of affording it changed
        a second variable: batch 8 halves the ranking term's within-batch pairs, fewer slices collides
        with the slice-count arm. Resampling moves the sampling grid and nothing else.

        The size is recorded in the member's config and fingerprint, because a member trained at 392
        from a 448 cache and rebuilt at 448 is a different function on different pixels — the same
        failure `slice_pooling` already caused once, caught then by the fingerprint gate.
        """
        self.read_size = size

    def exponential_average(self, decay: float = DECAY) -> None:
        """Keep an EMA of the weights, and score and save from it rather than from the live ones."""
        self.average = WeightAverage(self.model, decay)

    @contextmanager
    def evaluated(self) -> Iterator[None]:
        """Run a block under the weights this run would SAVE.

        Scoring the live weights while saving the averaged ones ranks epochs for a model that is never
        written to disk — the checkpoint's own number would then belong to a different function.
        """
        with self.average.applied(self.model):
            yield

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
            max_lr=[self.adaptation.head_lr, self.adaptation.backbone_lr],
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
        planned = len(positions) // max(size, 1)
        for batch in self.batches(positions, size, generator, shuffle=True):
            start = int(generator.integers(0, corpus.slices - self.window + 1))
            pixels = torch.from_numpy(np.ascontiguousarray(corpus.images[batch, :, start : start + self.window])).to(
                self.device
            )
            pixels = self.augment_bag(pixels)
            flags = torch.from_numpy(corpus.mask[batch]).to(self.device)
            score, weight = targets.rows(batch)
            with torch.autocast("cuda", enabled=self.device.type == "cuda"):
                # Both readouts from ONE encoder pass. The report head costs a small Linear and gets no
                # gradient when report supervision is off, so the loop needs no branch for it.
                logits, report = self.model.supervised(pixels, flags, self.read_size or corpus.image_size)
                loss = self.loss(
                    logits.float(),
                    torch.from_numpy(score).to(self.device),
                    torch.from_numpy(weight).to(self.device),
                ) + self.reports(report.float(), batch)
            self.optimiser.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimiser)
            self.scaler.update()
            if self.schedule is not None:
                self.schedule.step()
            self.average.update(self.model)
            total += float(loss.detach())
            steps += 1
            self.heartbeat.tick(steps, planned, total / steps, int(pixels.shape[0] * pixels.shape[1] * pixels.shape[2]))
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
        middle = (corpus.slices - self.window) // 2
        chunks = []
        with torch.no_grad(), torch.autocast("cuda", enabled=self.device.type == "cuda"):
            for batch in self.batches(positions, size, generator, shuffle=False):
                pixels = torch.from_numpy(
                    np.ascontiguousarray(corpus.images[batch, :, middle : middle + self.window])
                ).to(self.device)
                flags = torch.from_numpy(corpus.mask[batch]).to(self.device)
                chunks.append(
                    torch.sigmoid(self.model(pixels, flags, self.read_size or corpus.image_size).float()).cpu().numpy()
                )
        return np.concatenate(chunks).astype(np.float64)

    def loss(
        self,
        logits: Float[Tensor, "b o"],
        scores: Float[Tensor, "b o"],
        weights: Float[Tensor, "b o"],
    ) -> Float[Tensor, ""]:
        """The scalar this step descends, as decided by whichever objective this run holds."""
        return self.objective(logits, scores, weights)

    def save_partial(self, destination: Path, payload: dict[str, object]) -> None:
        """Write the best weights so far, atomically, the moment they improve.

        Holding the only copy in memory until the final epoch is how a run that has already produced
        its best epoch loses everything — which the OneCycle off-by-one demonstrated by raising on the
        very last step of a thirty-epoch run. Temp-then-rename means a crash mid-write cannot leave a
        truncated file behind either.

        The payload is assembled by the CALLER, and is the same one the final save writes. This used
        to store only weights, holdout and epoch, which preserved a run's work but not its identity:
        a member with no fingerprint and no recorded reading is refused by MemberPackage, so a killed
        run left weights that could be inspected and never submitted.
        """
        partial = destination.with_suffix(".partial")
        temporary = partial.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(partial)
