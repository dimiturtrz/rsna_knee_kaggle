"""MedicalNet/Med3D as a FROZEN VOLUMETRIC TRUNK — one 2048-wide vector per slot, read off layer4.

`arXiv:1904.00625`, Apache-2.0/MIT weights (their README declares MIT; the Kaggle mirror tags Apache
2.0 — either way permissive, no non-commercial clause and no share-alike), a 3D ResNet-50 pretrained on
the 23-dataset 3DSeg-8 aggregate of medical volumes. Nothing here fine-tunes it; this is an encoder that
turns a slot's twelve slices into one vector, so `feature_probe` can ask the question `ortho_diffusion`
left standing.

WHY THIS TRUNK, WHEN THE VOLUMETRIC AXIS ALREADY HAS A READING. It has exactly ONE, and that reading is
confounded by the file that produced it. `ortho_diffusion`'s own docstring pre-registered the confound:
"their middle block is 256 channels wide against dinov2-large's 1024… a ridge probe is width-sensitive,
so a LOSS here is confounded with capacity and only a WIN is clean." It then lost — `orthodiff@mean`
0.7435 solo, +0.0001 into `honest12`, against `ortholarge@mean` 0.8170 — which is the one branch that
file said it could not read. MedicalNet's layer4 is **2048** wide, WIDER than dinov2-large's 1024, so the
capacity explanation is removed rather than argued with. It is also the only clean-licensed distant asset
left standing after `knee-zkfin` closed every foreign MOUNT, and being a PRETRAINING rather than a mount
is what makes a number off it HONEST: we hold out our own 909, so it may enter the ledger, which no
raptor/pilkwang/coat number ever can.

PRE-REGISTERED BANDS, written before the build runs, against `ortholarge@mean` 0.8170 and
`orthodiff@mean` 0.7435 on the fold-0 holdout with each probe's own ridge penalty:

- **>= 0.8170** — the volumetric axis was closed on WIDTH, not on mechanism. It reopens as the one lane
  aimed at the 0.8696 solo the leader's board score implies, and the next spend is a FINE-TUNE of this
  trunk rather than another probe.
- **0.78 to 0.8169** — a real 3D representation that still trails our 2D trunks. Seat it for DIVERSITY
  only (`js` against the roster decides), and do not fund a fine-tune off it.
- **0.74 to 0.7799** — indistinguishable from `orthodiff` at eight times the width. Capacity was never
  the binding term, the volumetric axis is closed on MECHANISM, and this project stops paying for it.
- **< 0.74** — worse than a diffusion middle block at 8x the width, which is a HARNESS suspect before it
  is a result: check the intensity preparation and the depth resample before any verdict.

IMPORTED BY PATH, NEVER VENDORED — the rule `ortho_weights` follows for OrthoDiffusion and
`foreign_member` for raptor. Their checkout sits gitignored under `external/MedicalNet` and only
`models.resnet` is touched, which imports nothing outside torch.
"""

import logging
from pathlib import Path

import torch
from jaxtyping import Float, UInt8
from torch import Tensor, nn
from torch.nn import functional

from core.data.corpus.slots import SLOTS

logger = logging.getLogger(__name__)

EXTERNAL = Path("external/MedicalNet")
"""Their checkout, relative to the repo root. `git clone https://github.com/Tencent/MedicalNet`, pinned
at `20f76aaa`; gitignored per the external-deps rule."""

WEIGHTS = "medicalnet"
"""Their checkpoint directory under the data root's `external` kind, beside `orthodiffusion` and
`raptor`."""

CHECKPOINT = "resnet_50_23dataset.pth"
"""The 23-dataset pretraining rather than `resnet_50.pth`. Both are on disk and both are pure trunks
(318 tensors, no `conv_seg`); this one saw the widest corpus, which is the whole reason a pretraining is
being borrowed at all."""

READ_SIZE = 256
"""Matched to `ortho_diffusion.READ_SIZE` on purpose. Their network is fully convolutional and would
accept any size, so this is not a fact about the weights — it is what keeps the ONE thing that differs
between the two volumetric probes the trunk itself."""

DEPTH = 16
"""Matched to `ortho_diffusion.DEPTH`, same argument. Our pixel caches hold twelve slices per slot, so
the stack is interpolated UP rather than cropped."""

WIDTH = 2048
"""`layer4`'s output width — a ResNet-50 bottleneck's 512 planes times its expansion of 4. Asserted
against the probe's own output rather than trusted, because a silent mismatch here is a cache of the
wrong shape that loads perfectly."""

SEGMENTATION_CLASSES = 2
"""Their constructor demands a class count for the `conv_seg` head this trunk never runs and the
checkpoint does not carry. Any value builds the same encoder; it is passed because the signature is
theirs."""

SHORTCUT = "B"
"""Their projection shortcut. Type `A` is a zero-padded identity with NO parameters, so loading these
weights into it would leave every `downsample` tensor unmatched — the checkpoint names which one it is
(`module.layer1.0.downsample.0.weight` is a conv), and this records the reading."""

TRUNK_MODULES = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
"""Their `forward` in order, stopping one step short of it. The last thing their `forward` does is the
`conv_seg` head, so the encoder is named as the prefix rather than inherited — and naming the children
instead of reaching through attributes is also what keeps the walk typed, since a `Module` attribute is
a `Tensor | Module` to a checker and only `get_submodule` is a module by contract."""

LAST_STAGE = "layer4"
"""Where a classification ResNet is read, and where the first MedicalNet probe read (0.7348)."""

STAGE_WIDTHS = {"layer3": 1024, LAST_STAGE: WIDTH}
"""The readable stages and the width each one outputs. `layer3` is here because THEIR `layer3` runs at
dilation 2 and `layer4` at dilation 4 — dilations a SEGMENTATION network wants and a whole-volume
readout may not. That makes the depth of the read a named harness suspect against the 0.7348 band
rather than a knob: if the deepest features are the segmentation-specialised ones, the stage below them
is where a classification signal would still be, and it is the one alternative that could reopen the
axis. A stage shallower than `layer3` is not offered — at that depth the read is texture."""


class MedicalNetTrunk(nn.Module):
    """One plane-agnostic 3D ResNet-50, read as an encoder over our six-slot pixel cache.

    Plane-agnostic where `OrthoDiffusionTrunk` is per-plane, and that is a fact about the asset rather
    than a simplification: they published ONE network trained across eight segmentation corpora in every
    orientation, not three orientation-specific ones.
    """

    def __init__(self, resnet: nn.Module, stage: str = LAST_STAGE) -> None:
        super().__init__()
        self.resnet = resnet
        self.walk = TRUNK_MODULES[: TRUNK_MODULES.index(stage) + 1]

    @staticmethod
    def prepared(volume: UInt8[Tensor, "b g p q"]) -> Float[Tensor, "b c d h w"]:
        """One slot's stack as their loader would have handed it over: z-scored, depth-first, one channel.

        Their `__itensity_normalize_one_volume__` z-scores over the NON-ZERO voxels of a volume, not over
        all of them, because a segmentation volume is mostly background and including it would drag every
        real intensity into a narrow band. Copied rather than replaced by the [-1, 1] min-max
        `OrthoDiffusionTrunk` uses, because a frozen trunk reads whatever distribution it was trained on
        and the preparation is part of the weights.

        A slot that was never acquired is all zeros in the pixel cache and has no non-zero voxels at all;
        the epsilon keeps it finite, and the presence mask beside the features is what tells a head to
        ignore it.
        """
        stack = volume.unsqueeze(1).float()
        resampled = functional.interpolate(
            stack, size=(DEPTH, READ_SIZE, READ_SIZE), mode="trilinear", align_corners=False
        )
        present = (resampled > 0).float()
        count = present.sum(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1.0)
        mean = (resampled * present).sum(dim=(1, 2, 3, 4), keepdim=True) / count
        variance = (((resampled - mean) * present) ** 2).sum(dim=(1, 2, 3, 4), keepdim=True) / count
        return (resampled - mean) / (variance.sqrt() + torch.finfo(torch.float32).eps) * present

    @staticmethod
    def pooled(activation: Float[Tensor, "b c d h w"]) -> Float[Tensor, "b c"]:
        """Global average over the three spatial axes — the readout their own linear probes use."""
        return activation.mean(dim=(2, 3, 4))

    def encoded(self, volume: UInt8[Tensor, "b g p q"]) -> Float[Tensor, "b w"]:
        """One slot through the trunk, read at `layer4` and pooled.

        The stages are walked here rather than through their `forward`, which ends in the `conv_seg`
        transpose-convolution head — a head this checkpoint does not carry and a probe does not want.
        """
        read = self.prepared(volume)
        for name in self.walk:
            read = self.resnet.get_submodule(name)(read)
        return self.pooled(read)

    def slice_features(
        self, pixels: UInt8[Tensor, "b s g p q"], read_size: int | None = None
    ) -> Float[Tensor, "b s g w"]:
        """Every slot of a batch as one vector each — the `SliceFeatures` shape `EmbeddingCache` builds by.

        The depth axis of the result is 1 and not `g`: the slices are consumed INSIDE the encoder, which
        is the whole-stack case `EmbeddingSpec.slice_read` exists to name and `_probe_shape` already
        probes for rather than assuming.

        `read_size` is accepted and ignored, for the same reason `OrthoDiffusionTrunk` ignores it: the
        size this trunk reads at is pinned to the sibling probe so the comparison has one free variable,
        and silently honouring a different one would break exactly that.
        """
        del read_size
        encoded = [self.encoded(pixels[:, index]) for index, _ in enumerate(SLOTS)]
        return torch.stack(encoded, dim=1).unsqueeze(2)
