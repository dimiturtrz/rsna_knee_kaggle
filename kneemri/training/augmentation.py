"""What a training image is perturbed by before the model sees it, and how hard.

ITS OWN MODULE, not a block of `Trainer`. The perturbations are a pure function of pixels and a
generator — no model, no optimiser, no device budget — and they were the largest thing in a file that
also owns the optimiser, the schedule, the epoch loop and the checkpoint writer. Testing a gamma curve
had to build a `StudyModel` first, and `trainer.py` sat at 666 lines against a 750-line god-file gate
with the augmentation block the obvious two hundred that do not belong to fitting.

The RNG order is the one thing this move had to preserve: a member's pixels are decided by the sequence
of draws, so the transforms stay in their old order, drawing exactly what they drew before.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, UInt8
from torch import Tensor, nn

from kneemri.models.drop_path import NO_DROP
from kneemri.models.slot_dropout import SlotDropout
from kneemri.training.config_leaf import ConfigLeaf
from kneemri.training.ghosting import Ghosting
from kneemri.training.mixup import OFF
from kneemri.training.progress_schedule import FULL, ProgressSchedule

AUG_SUBJECT = (
    "the ONE progress schedule scaling every regulariser at once (geometric and intensity widths, ghosting, "
    "slot dropout, mixup alpha, stochastic depth)"
)

PATCH_ERASE = "patch_erase"
"""The key patch erasing answers under when a run checks its switches landed (`kneemri.training.landed`).

A rate, and it is read off the `PatchErasing` the loop calls rather than off the config: the maps are
loaded from a file that may name none of this corpus's studies, and a rate that reached no map is the
failure this key exists to show.
"""

SLOT_DROPOUT = "slot_dropout"
"""The key slot dropout answers under when a run checks its switches landed (`kneemri.training.landed`).

A RATE RATHER THAN A FLAG, and it is reported from the config the augmenter will actually use. The
failure this guards is the one `run_setup` makes reachable: the rate is threaded onto an `AugmentConfig`
built by one of two branches, so a branch that forgets it trains the geometric recipe under the arm's
name and reports a number about a run nobody asked for.
"""

SCALE_JITTER = "scale_jitter"
"""The key field-of-view jitter answers under when a run checks its switches landed.

A WIDTH RATHER THAN A FLAG, and its identity is 0.08 rather than zero — the value every member on disk
was fitted under, hardcoded until knee-su2sw. That is what makes it worth reporting: an arm asking for a
wider field can only be told apart from the whole pool by the number, and a branch in `run_setup` that
dropped it would train the historical recipe under the arm's name and report a null about a mechanism
that never ran.
"""

PADDING_MODE = "padding_mode"
"""The key the resample's out-of-frame rule answers under when a run checks its switches landed.

A NAME RATHER THAN A FLAG, and its identity is "border" — what `_warp` hardcoded until knee-10f10, so it
is the rule every member in the pool was fitted under. Worth reporting for the same reason
`SCALE_JITTER` is: the rule only has an effect where the sampling grid leaves the cached crop, which is
exactly the region a widened jitter spends its draws in, so an extent arm and a padding arm are
indistinguishable by anything except these two numbers.
"""

INTENSITY_AUGMENT = "intensity_aug"
"""The key an augmentation answers under when a run checks its switches landed (`kneemri.training.landed`).

The intensity block is chosen by which CONFIG the trainer was constructed with, not by a method a
caller may forget to call — so its absence is `geometric_only`, a perfectly good recipe that is the one
every earlier member was fitted under and the one an intensity arm would be silently compared against.
"""


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
    """Half-width of the uniform FIELD-OF-VIEW jitter: `_warp` samples a scale in `1 +/- this`.

    Above 1 the sampling grid runs past the cached crop and `padding_mode="border"` replicates edge
    pixels, so a zoom-OUT fabricates margin rather than revealing tissue. The cache's own `--crop-mm` is
    therefore what decides how much of this range is real, and a wide jitter belongs on a wide cache.

    0.08 is the public baseline's and was hardcoded until knee-su2sw, which is to say every member in
    the pool saw the extent vary by +/-8% about a 130 mm crop and none of them ever saw it vary more.
    D206 section 5 measured the foreign corpus's span 20-50% wider than ours, and the 11-member readout
    of 2026-09-09 found the generalisation toll FLAT across the pool (-0.0336 +/- 0.0038) with the one
    widened-field member paying least. A blind spot every member shares cannot show up as variance
    between them, so it is the cheapest thing here that a solo offline ruler cannot see.
    """

    padding_mode: str = "border"
    """What `_warp`'s resample reads where its sampling grid leaves the image — `grid_sample`'s own
    vocabulary, so "zeros", "border" or "reflection".

    "border" was hardcoded until knee-10f10 and is therefore the rule every member on disk was fitted
    under. Its stated reason was rotation: a rotated knee should run out into tissue-like values rather
    than into a black wedge the encoder can use to infer how far the image was turned. That argument is
    sound for an 8-degree rotation, which leaves only a thin corner out of frame. It was never an
    argument about EXTENT, and above scale 1 the whole margin is out of frame, so a zoom-OUT fabricates
    tissue by replicating one edge row across it.

    The literature declines to make this choice at all (D258, 2026-09-09): RandomResizedCrop crops
    inside the image so no border ever appears (MAE, DINO), nnU-Net oversamples its patch by the scale
    envelope so a zoomed-out sample comes from real tissue, and Chen 2020 states "random cropping after
    all other image augmentations, without black borders". torchvision and MONAI pad with constant or
    zeros, TorchIO with an Otsu background estimate. Reflection is offered everywhere and NO
    medical-imaging paper reports it beating crop-inside. So there is no published number to
    pre-register against here — the ordering reflection > border on a zoom-out past the cached field is
    inference from general practice, and measuring it is a first read rather than a replication.
    """

    shift_jitter: float = 0.05
    gain_jitter: float = 0.10
    gamma: tuple[float, float] = (0.7, 1.5)
    gamma_probability: float = 0.3
    blur_probability: float = 0.2
    contrast: tuple[float, float] = (0.85, 1.15)
    noise: float = 0.02
    """Noise is far below the sibling's 0.08: theirs acts on z-scored images, ours on [0, 1]."""

    slot_dropout: float = 0.0
    """Probability that a PRESENT slot is withheld from a training study — modality dropout.

    Not an intensity op and not a geometric one: it perturbs which ACQUISITIONS the study is made of,
    which is the axis this corpus varies on and no member has ever been trained against. Fill rates are
    100% for the axial fluid-weighted slot but 55-60% for sagittal non-fat-sat and both T1 slots, so
    nearly half of all studies reach inference with slots the model has only ever seen filled. The head
    already MASKS an absent slot rather than zero-filling it (`SlotScheme.presence`), so the gap is not
    that absence is misrepresented — it is that absence was never in the training distribution.

    Zero by default, which is the identity and the recipe every member on disk was fitted under.

    PRECEDENT, AT ITS REAL STRENGTH. Grovik et al. 2021 (npj Digital Medicine) report input-level
    dropout holding AUC exactly flat (0.989 against 0.989) while Dice rose 0.021 and false positives
    per patient fell from 7.0 to 3.6. Chrysochoou et al. 2026 (npj Precision Oncology) report dropout
    beating both zero-fill and generative synthesis on every missing-sequence cell. Both are
    SEGMENTATION, and the one paper that measures our metric measured no gain on it — so this is not
    armed as a solo-score lever. It is armed because it changes what evidence a member may rely on,
    which is a supervision axis rather than a seating one.
    """

    ghosting: float = 0.0
    """Target median per-pixel residual for phase-encode motion ghosting, in [0, 1] image units.

    Zero is the identity and the recipe every member on disk was fitted under. The value KMAR-50K's
    paired volumes measure is **0.040** — see `kneemri.training.ghosting`, which owns the mechanism and
    the argument for why this field is a measured residual rather than a replica amplitude.

    It is the only augmentation in this config whose magnitude was READ OFF A CORPUS rather than chosen.
    That is the whole reason it exists: `slot_dropout` is the one field of the three shared blind spots
    that survived 2026-09-12 (cross-site toll -0.0271 against its control's -0.0361, +0.0090 shallower
    and 2.4 sd of the eleven-member pool), and it is a corruption-robustness operator dialled to taste.
    This is the same mechanism class with the taste removed.
    """

    @classmethod
    def geometric_only(cls) -> "AugmentConfig":
        """The public baseline's augmentation: rigid jitter and one global gain, no intensity ops.

        This is the DEFAULT, so that enabling the intensity block is an explicit, ablatable decision
        rather than a silent change of recipe. It also keeps a run reproducible against members that
        were fitted before the block existed.
        """
        return cls(gamma_probability=0.0, blur_probability=0.0, contrast=(1.0, 1.0), noise=0.0)

    def intensity(self) -> bool:
        """Whether the intensity block does ANYTHING, read off the widths rather than off a flag.

        `geometric_only` disables it by zeroing every width, so there is no boolean to consult — and a
        config assembled with the intensity fields left at their geometric values is the same recipe
        whatever the run's flag said. Asking the widths is asking what the batch will actually get.
        """
        return bool(self.gamma_probability or self.blur_probability or self.noise or self.contrast != (1.0, 1.0))

    def scaled(self, strength: float) -> "AugmentConfig":
        """Every width at this multiple, the one knob a `ProgressSchedule` turns — `self` at exactly 1.0.

        ONE MULTIPLIER FOR EVERY OP, no per-op schedule: widths scale about their identity (0 for a
        jitter, 1 for a gamma or contrast factor), probabilities and slot dropout clip at 1. Returning
        `self` at the identity is what keeps the default loop bit-identical, since `1 + (0.7 - 1) * 1`
        is not the float 0.7.
        """
        if strength == FULL:
            return self

        def about_one(pair: tuple[float, float]) -> tuple[float, float]:
            return (FULL + (pair[0] - FULL) * strength, FULL + (pair[1] - FULL) * strength)

        return replace(
            self,
            rotation_degrees=self.rotation_degrees * strength,
            scale_jitter=self.scale_jitter * strength,
            shift_jitter=self.shift_jitter * strength,
            gain_jitter=self.gain_jitter * strength,
            gamma=about_one(self.gamma),
            gamma_probability=min(self.gamma_probability * strength, FULL),
            blur_probability=min(self.blur_probability * strength, FULL),
            contrast=about_one(self.contrast),
            noise=self.noise * strength,
            slot_dropout=min(self.slot_dropout * strength, FULL),
            ghosting=self.ghosting * strength,
        )


class Augmentation:
    """The perturbations one config asks for, applied to a batch of images or a batch of slot bags."""

    _BLUR = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
    """A 3x3 Gaussian, normalised so a blur cannot change the image's overall level."""

    def __init__(self, config: AugmentConfig | None = None) -> None:
        self.base = config or AugmentConfig.geometric_only()
        """The config the run ASKED for — what a landed-check compares, whatever strength is current."""
        self.config = self.base
        """The config this step applies: `base` scaled by the progress schedule's current strength."""

    def intensity(self) -> bool:
        """Whether this augmentation perturbs intensity, asked of the config the run asked for."""
        return self.base.intensity()

    def strengthen(self, strength: float) -> None:
        """Apply `base` at this multiple from the next batch on — see `AugmentConfig.scaled`."""
        self.config = self.base.scaled(strength)

    def applied(self, images: Float[Tensor, "n c h w"], generator: torch.Generator) -> Float[Tensor, "n c h w"]:
        """Rigid jitter, then intensity — never a flip, which laterality normalisation forbids.

        Every transform is per-sample and vectorised: parameters come from one `torch.rand(n, ...)` and
        probabilistic transforms are applied with a mask (`out = do*transformed + (1-do)*original`)
        rather than a Python branch, so a batch costs one kernel per transform and not one per image.
        """
        warped = self._warp(images, generator)
        # Ghosting LAST, after the intensity block. It is an acquisition artefact, so it belongs on the
        # image the scanner would have produced — putting it before a gamma curve would have the gamma
        # act on the replicas too, which no scanner does and which would break the residual the operator
        # was calibrated to produce.
        return self._ghost(self._intensity(warped, generator), generator)

    def _ghost(self, images: Float[Tensor, "n c h w"], generator: torch.Generator) -> Float[Tensor, "n c h w"]:
        """Phase-encode motion replicas at the config's measured severity, or the identity at zero."""
        return Ghosting(self.config.ghosting).applied(images, generator)

    def _warp(self, images: Float[Tensor, "n c h w"], generator: torch.Generator) -> Float[Tensor, "n c h w"]:
        """Rotation, scale and translation as a single affine resample."""
        config, count, device = self.config, images.shape[0], images.device
        angles = (torch.rand(count, generator=generator, device=device) * 2 - 1) * np.deg2rad(config.rotation_degrees)
        scales = 1 + (torch.rand(count, generator=generator, device=device) * 2 - 1) * config.scale_jitter
        shifts = (torch.rand(count, 2, generator=generator, device=device) * 2 - 1) * config.shift_jitter
        cosine, sine = torch.cos(angles) / scales, torch.sin(angles) / scales
        theta = torch.zeros(count, 2, 3, device=device)
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cosine, -sine, shifts[:, 0]
        theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sine, cosine, shifts[:, 1]
        grid = nn.functional.affine_grid(theta, list(images.shape), align_corners=False)
        return nn.functional.grid_sample(images, grid, align_corners=False, padding_mode=config.padding_mode)

    def _intensity(self, images: Float[Tensor, "n c h w"], generator: torch.Generator) -> Float[Tensor, "n c h w"]:
        """Gain, gamma, blur, contrast and noise — the scanner-variation axis, on [0, 1] inputs.

        EACH OP IS SKIPPED WHEN ITS OWN CONFIG DISABLES IT, which `noise` already did and the other
        three did not. A disabled op is not free: the mask arithmetic multiplies its result by zero,
        but the result is computed first, over the whole batch. The default config is
        `geometric_only`, so every step of every run in the ledger has paid for a full-tensor `pow`
        and a `conv2d` blur whose outputs were then discarded.

        MEASURED on a 5090 at the whole-stack shape (batch 12, six slots, twelve slices, 448 px):
        15.5 ms a step against 2.0 ms, and 3.23 GB of transient allocation against 1.29. The time is
        ~1.2% of a ~1.15 s step — directional, not a result; the allocation is freed before the
        encoder runs, so it does not move the run's peak either. It is taken because computing a
        value in order to multiply it by zero is dead work, not because either number is a lever.

        The enabled path is unchanged (18.7 ms against 19.0 — one guard's worth of noise). The
        disabled path differs by one ulp: dropping `(x - mean) * 1 + mean` is not bitwise identity,
        and after requantisation to uint8 that shows as a 1 LSB difference on 1e-6 of pixels, at
        rounding boundaries only.

        A SKIPPED OP ALSO SKIPS ITS DRAWS, so the generator advances differently and a rerun of an
        existing arm gets a different realisation of the same augmentation distribution. That is a
        reproducibility cost with no distributional one, and it is the reason this is worth naming
        rather than filing as a tidy-up: the pixels differ, the recipe does not.
        """
        config, count, device = self.config, images.shape[0], images.device
        shape = (count, 1, 1, 1)

        gain = 1 + (torch.rand(shape, generator=generator, device=device) * 2 - 1) * config.gain_jitter
        images = images * gain

        if config.gamma_probability > 0:
            low, high = config.gamma
            exponent = torch.rand(shape, generator=generator, device=device) * (high - low) + low
            apply_gamma = (torch.rand(shape, generator=generator, device=device) < config.gamma_probability).float()
            images = apply_gamma * images.clamp(0, 1) ** exponent + (1 - apply_gamma) * images

        if config.blur_probability > 0:
            apply_blur = (torch.rand(shape, generator=generator, device=device) < config.blur_probability).float()
            channels = images.shape[1]
            kernel = self._BLUR.to(device, images.dtype).view(1, 1, 3, 3).expand(channels, 1, 3, 3)
            blurred = nn.functional.conv2d(images, kernel, padding=1, groups=channels)
            images = apply_blur * blurred + (1 - apply_blur) * images

        low, high = config.contrast
        if (low, high) != (1.0, 1.0):
            contrast = torch.rand(shape, generator=generator, device=device) * (high - low) + low
            # Contrast scales about the image's own mean, so it changes the SPREAD of intensities without
            # also shifting the level — which `gain` above is already responsible for.
            mean = images.mean(dim=(1, 2, 3), keepdim=True)
            images = (images - mean) * contrast + mean

        if config.noise > 0:
            images = images + torch.randn(images.shape, generator=generator, device=device) * config.noise
        return images.clamp(0, 1)

    def dropped(
        self,
        images: UInt8[Tensor, "b n g h w"],
        presence: Float[Tensor, "b n"],
        generator: torch.Generator,
    ) -> tuple[UInt8[Tensor, "b n g h w"], Float[Tensor, "b n"]]:
        """Withhold present slots at random — the rule `SlotDropout` owns, at this config's rate."""
        return SlotDropout(self.config.slot_dropout).dropped(images, presence, generator)

    def bagged(self, images: UInt8[Tensor, "b n g h w"], generator: torch.Generator) -> UInt8[Tensor, "b n g h w"]:
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
        warped = self.applied(flat, generator)
        return (warped * 255).round().clamp(0, 255).to(torch.uint8).reshape(images.shape)


PADDING_MODES = ("zeros", "border", "reflection")
"""What `grid_sample` fills outside the cached crop with."""


@dataclass(frozen=True)
class RegularisationConfig:
    """The regularisers a run arms and the progress schedule that scales them."""

    mixup: float = ConfigLeaf.of(
        default=OFF,
        says=(
            "blend each batch with a permutation of itself, pixels and TARGETS together, at a weight drawn "
            "from Beta(a, a). 0.1-0.4 is the usual range; 0, the default, is off and is what every ledger "
            "number was measured under. This is the first label-space regulariser here: --intensity-aug "
            "changes how a study looks and leaves its label alone, which is why it fired and the overfitting "
            "did not care - band_dense_long peaked at epoch 6 and lost 0.0182 over eighteen more while its "
            "train loss halved."
        ),
    )
    slot_dropout: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "withhold each PRESENT slot from a training study with this probability, pixels and presence mask"
            " together. 0, the default, is off and is what every ledger number was measured under. The first "
            "SUPERVISION-space regulariser here: --mixup perturbs the label and --intensity-aug perturbs the "
            "appearance, and neither changes which acquisitions a study is made of. This corpus varies on "
            "exactly that axis - the axial fluid slot fills 100%% of studies while both T1 slots fill 55-60%%"
            " - so nearly half of all studies reach inference with slots the model has only ever seen filled."
            " A study is never emptied."
        ),
    )
    patch_maps: Path | None = ConfigLeaf.of(
        default=None,
        says=(
            "npz of an occlusion sweep over the TRAINING studies (keys 'studies' and 'maps'), written by "
            "kneemri.evaluation.patch_maps --split train. Read by --patch-erase and by nothing else."
        ),
    )
    patch_erase: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "blank one patch of each training study with this probability, the patch drawn in proportion to "
            "what --patch-maps priced that study's cells at. 0, the default, is off. It removes the evidence "
            "the swept member actually leaned on, so the head has to reach the redundant evidence instead - "
            "centre-only and border-only readouts of one member sit at 0.8021 and 0.8006, so that redundancy "
            "is there to reach. The map's ANATOMY is not relied on and is known to be wrong (knee-j19z1): a "
            "map marks what a model reads, which is all an augmentation consumes."
        ),
    )
    ghosting: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "add phase-encode motion ghosting to each training image at this TARGET MEDIAN per-pixel "
            "residual, in [0, 1] image units. 0, the default, is off and is what every ledger number was "
            "measured under. The only augmentation here whose magnitude is measured rather than chosen: "
            "KMAR-50K ships the same knee acquired with and without patient motion, and 109 of its pairs put "
            "the residual at 0.040 of full scale (p10 0.023, p90 0.086), the impulsiveness at p99/mean ~ 9 "
            "and 11%% of the residual energy outside the anatomy - the ghosting signature, which is why this "
            "displaces replicas rather than blurring."
        ),
    )
    scale_jitter: float = ConfigLeaf.of(
        default=0.08,
        says=(
            "half-width of the uniform FIELD-OF-VIEW jitter: each training study is resampled at a scale "
            "drawn from 1 +/- this. 0.08, the default, is the public baseline's and is what every ledger "
            "number was measured under - it was hardcoded until knee-su2sw, so no member on disk has ever "
            "seen the extent vary by more. The foreign corpus's span runs 20-50%% wider (D206), and the "
            "11-member readout of 2026-09-09 measured the generalisation toll FLAT across the pool at -0.0336"
            " +/- 0.0038 with the single widened-field member paying least - a blind spot every member shares"
            " cannot appear as variance between them. Above 1 the grid runs past the cached crop and border "
            "pixels are replicated, so a wide jitter wants a wide --crop-mm under it or half its draws "
            "fabricate margin instead of revealing tissue."
        ),
    )
    padding_mode: str = ConfigLeaf.of(
        default="border",
        says=(
            "what the augmentation resample reads where its grid leaves the image. 'border', the default, "
            "replicates the edge row and is what every ledger number was measured under - it was hardcoded "
            "until knee-10f10. Its reason was ROTATION (a turned knee should run out into tissue-like values,"
            " not a black wedge that says how far it was turned), which is sound for 8 degrees and was never "
            "an argument about EXTENT: above scale 1 the entire margin is out of frame, so a zoom-out "
            "fabricates tissue from one replicated row. The literature declines the choice instead of making "
            "it - MAE and DINO crop INSIDE the image, nnU-Net oversamples its patch by the scale envelope so "
            "a zoomed-out sample is real tissue - and no medical-imaging paper reports reflection beating "
            "crop-inside, so this is a first read rather than a replication (D258)."
        ),
        choices=PADDING_MODES,
    )
    intensity_aug: bool = ConfigLeaf.of(default=False, says="gamma/blur/contrast/noise too")
    stochastic_depth: float = ConfigLeaf.of(
        default=NO_DROP,
        says=(
            "rate at which the trunk's DEEPEST block drops its whole residual branch for a sample during "
            "training, ramped linearly from zero at the first block (0 = the intact trunk every number in the"
            " ledger was measured under). It costs nothing -- no second forward, no parameter, no buffer -- "
            "and it is identity in eval(), so a member trained under it serves byte-identically. Same axis as"
            " --sharpness and a different mechanism: that one flattens the minimum the walk settles in, this "
            "one stops any single block's branch from being depended on. transformers applies a FLAT rate at "
            "every depth, which is not the schedule the published effect sizes were measured under; this "
            "ramps."
        ),
    )
    aug_start: float = ConfigLeaf.of(
        default=1.0,
        says=ProgressSchedule.knob_help("start", AUG_SUBJECT),
    )
    aug_mid: float = ConfigLeaf.of(
        default=1.0,
        says=ProgressSchedule.knob_help("mid", AUG_SUBJECT),
    )
    aug_end: float = ConfigLeaf.of(
        default=1.0,
        says=ProgressSchedule.knob_help("end", AUG_SUBJECT),
    )
    aug_ramp: float = ConfigLeaf.of(
        default=0.0,
        says=ProgressSchedule.knob_help("ramp", AUG_SUBJECT),
    )
    aug_tail: float = ConfigLeaf.of(
        default=0.0,
        says=ProgressSchedule.knob_help("tail", AUG_SUBJECT),
    )
