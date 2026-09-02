"""OrthoDiffusion as a FROZEN VOLUMETRIC TRUNK — one vector per slot, read off a denoiser's middle block.

`arXiv:2602.20752`, MIT-licensed weights, three plane-specific 3D U-Nets trained as unconditional
diffusion models over musculoskeletal MRI. Nothing here fine-tunes them; this is an encoder that turns a
slot's twelve slices into one 256-wide vector, so `feature_probe` can ask the only question worth asking
first: does a 3D in-domain representation carry more of the twelve findings than our 2D trunks do?

WHY THIS TRUNK AND NOT ANOTHER ONE. The standing diagnosis on the ledger is that the slice-AGGREGATION
stage is saturated from both ends — max-pool, attention, moments and focal readouts all land inside the
noise floor of each other — and that "the limit is upstream, in what one slice's feature vector
contains". Every trunk this project has measured encodes ONE SLICE at a time and leaves the third
dimension to a pooling. This one consumes the stack inside the encoder, which is the axis the diagnosis
names and the only open lever whose mechanism could plausibly move 1.5%.

WHAT THE PROBE CANNOT SEPARATE, AND MUST SAY SO. Their middle block is 256 channels wide against
dinov2-large's 1024, and after the three planes are read the widths are still not matched. A ridge probe
is width-sensitive, so a LOSS here is confounded with capacity and only a WIN is clean. That is the
right asymmetry for a screening instrument and the wrong one for a verdict.

IMPORTED BY PATH, NEVER VENDORED — the same rule `foreign_member` follows for raptor. Their checkout
sits gitignored under `external/OrthoDiffusion` and only `diffusion_model.unet` is touched, which imports
nothing outside torch and numpy. Their `trainer.py` (einops, nibabel, tensorboard, torchio) is the reason
`diffusion_schedule` reimplements the six lines of `q_sample` rather than importing them.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from jaxtyping import Float, UInt8
from torch import Tensor, nn
from torch.nn import functional

from core.data.corpus.slots import SLOTS, Plane
from kneemri.models.diffusion_schedule import DiffusionSchedule

logger = logging.getLogger(__name__)

EXTERNAL = Path("external/OrthoDiffusion")
"""Their checkout, relative to the repo root. `git clone https://github.com/lt-0123/OrthoDiffusion`,
pinned at `99d1518`; gitignored per the external-deps rule."""

WEIGHTS = "orthodiffusion"
"""Their three 553 MB checkpoints, under the data root's `external` kind beside raptor's. License `mit`,
declared on the HuggingFace card — the Kaggle mirror that Kimi flagged as licence-unknown is the same
weights without the card."""

READ_SIZE = 256
"""The spatial size their U-Nets were trained at. `create_model` derives its channel multipliers from
this, so it is not a resampling preference — another value builds a different network."""

DEPTH = 16
"""Their `depth_size`. Our pixel caches hold twelve slices per slot, so the stack is INTERPOLATED up
rather than centre-cropped down: their loader only ever had to remove slices, and had no branch for a
volume thinner than the model's depth."""

CHANNELS = 64
"""Their `num_channels`, the base width the multipliers scale."""

RESIDUAL_BLOCKS = 1
"""Their `num_res_blocks`."""

MIDDLE_WIDTH = 256
"""Width of the middle block, `CHANNELS` times the last channel multiplier (4) at 256 px. Asserted
against the probe's own output rather than trusted, because a silent mismatch here is a cache of the
wrong shape that loads fine."""

NOISE_SEED = 2026
"""The generator seed. Their sampler draws fresh noise per call, which for a CACHED feature array means
the file describes one arbitrary draw nobody wrote down. Fixed so an encode is reproducible."""


@dataclass(frozen=True)
class PlaneRead:
    """Where one plane's trunk is read: its weights, the block, and the timestep it is noised to.

    Per-plane rather than global because their linear-probe sweep does not land on one setting for all
    three — the paper's LP-Opt reads the axial and sagittal models early and shallow and the coronal one
    later and deeper. Grouped so a caller cannot pair a coronal checkpoint with a sagittal timestep.
    """

    checkpoint: str
    block: str
    timestep: int


PLANE_READS: dict[str, PlaneRead] = {
    Plane.SAGITTAL: PlaneRead("sagittal_model.pt", "mid_0", 50),
    Plane.CORONAL: PlaneRead("coronal_model.pt", "mid_2", 100),
    Plane.AXIAL: PlaneRead("axial_model.pt", "mid_0", 50),
}
"""One read per plane, keyed by the plane STRING our SLOTS already carry — `Plane` is a `StrEnum` and
`Slot.plane` is typed as the string, so keying by the enum would be a second spelling of one key. A
declaration, not a bucket: every entry names a checkpoint that exists on disk, and a plane with no entry
cannot be encoded at all."""


class OrthoDiffusionTrunk(nn.Module):
    """Three plane-specific denoisers, read as one encoder over our six-slot pixel cache."""

    def __init__(self, denoisers: dict[str, nn.Module], schedule: DiffusionSchedule | None = None) -> None:
        super().__init__()
        self.denoisers = nn.ModuleDict(dict(denoisers))
        self.schedule = schedule or DiffusionSchedule()

    @staticmethod
    def prepared(volume: UInt8[Tensor, "b g p q"]) -> Float[Tensor, "b c d h w"]:
        """One slot's stack as their loader would have handed it over: [-1, 1], depth-first, one channel.

        Min-max per VOLUME rather than per slice, which is what their `Compose` does — a per-slice
        normalisation would erase the intensity relationship between slices, and that relationship is
        most of what a volumetric encoder is being asked to read.

        A slot that was never acquired is all zeros in the pixel cache and stays a constant volume here;
        the epsilon keeps it from dividing by zero, and the presence mask beside the features is what
        tells a head to ignore it.
        """
        stack = volume.unsqueeze(1).float()
        resampled = functional.interpolate(
            stack, size=(DEPTH, READ_SIZE, READ_SIZE), mode="trilinear", align_corners=False
        )
        lowest = resampled.amin(dim=(1, 2, 3, 4), keepdim=True)
        highest = resampled.amax(dim=(1, 2, 3, 4), keepdim=True)
        scaled = (resampled - lowest) / (highest - lowest + torch.finfo(torch.float32).eps)
        return scaled * 2.0 - 1.0

    @staticmethod
    def pooled(activation: Float[Tensor, "b c d h w"]) -> Float[Tensor, "b c"]:
        """Global average over the three spatial axes.

        GAP and not their `SelfAttnPooling3D`, which is a LEARNED module: untrained it is a random
        projection, and a probe read through one would be measuring the initialisation. Their own paper
        reports GAP as the pooling their linear probes use for exactly this reason.
        """
        return activation.mean(dim=(2, 3, 4))

    def encoded(self, volume: UInt8[Tensor, "b g p q"], plane: str) -> Float[Tensor, "b w"]:
        """One slot through its plane's denoiser, read at that plane's block and timestep."""
        read = PLANE_READS[plane]
        prepared = self.prepared(volume)
        generator = torch.Generator(device=prepared.device).manual_seed(NOISE_SEED)
        noisy = self.schedule.noised(prepared, read.timestep, generator)
        steps = torch.full((len(noisy),), read.timestep, dtype=torch.long, device=prepared.device)
        _, info = self.denoisers[plane](noisy, steps, ret_mid=True)
        return self.pooled(cast("Float[Tensor, 'b c d h w']", info["mid"][read.block]))

    def slice_features(
        self, pixels: UInt8[Tensor, "b s g p q"], read_size: int | None = None
    ) -> Float[Tensor, "b s g w"]:
        """Every slot of a batch as one vector each — the `SliceFeatures` shape `EmbeddingCache` builds by.

        The depth axis of the result is 1 and not `g`: the slices are consumed INSIDE the encoder, which
        is the whole-stack case `EmbeddingSpec.slice_read` exists to name and `_probe_shape` already
        probes for rather than assuming.

        `read_size` is accepted and ignored. Their channel multipliers are derived from 256, so a read at
        any other size is a different network and not a different reading of this one — the argument is
        part of the callable's contract and refusing it silently would be worse than saying so here.
        """
        del read_size
        encoded = [self.encoded(pixels[:, index], slot.plane) for index, slot in enumerate(SLOTS)]
        return torch.stack(encoded, dim=1).unsqueeze(2)
