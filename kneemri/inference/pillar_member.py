"""The Pillar-0 knee fine-tune, rebuilt from its own archive and read over our studies as a volume.

WHY IT EXISTS, AND WHY IT IS NOT `foreign_member`. Every member this pool owns — ours and the seated
foreign ones alike — is a 2D encoder run per slice with a pooling operator over the stack. The pinned
diagnosis of the plateau is that the pool is an ARCHITECTURE MONOCULTURE. `fabotelli/pillar0-knee-finetune`
is the only publicly-licensed knee checkpoint found that is natively volumetric: one 3D trunk consuming a
whole series at `[192,384,384]`, no per-slice step and no fusion operator anywhere in it. That is the
axis, and it is the reason to spend anything here — NOT its gold-58 macro-AUC of 0.797, which is below
every lane ceiling this project already owns.

WHAT IT CANNOT REUSE. `ForeignMember.rebuild` loads `weights_only=True` because the argument for running a
stranger's checkpoint is that nothing inside it ever executes. That argument does not survive here: the
Atlas trunk is not a `timm` name, so the CLASS has to come from the publisher's source tree and it does
run. The trade is stated rather than hidden — the weights are still loaded `weights_only=True`, so the
pickle still never executes; only the published, pinned, reviewable module does. That import, and the
`sys.modules` stubbing its chain needs, live in `atlas_import` — a different subject from a member.

THE FEED IS THE RISK AND IT IS BIGGER HERE THAN IT WAS FOR RAPTOR. Their adapter reads a full series:
`uint8 [D,256,256]` for a real D. Our cache holds twelve slices per slot, so the depth axis handed to a
trunk that expects a resolved volume is mostly interpolation. A low reading is therefore ambiguous between
the model and the feed in exactly the way `foreign_gate` exists to separate, and it should be pointed at
this key before any number from here is quoted as a fact about Pillar-0.

THE PERMUTATION HAS A FREE ORACLE, SO USE IT. `PILLAR_LABELS` is transcribed from the archive's README
table, which is a REPORT and not proof of the head's column order; a permuted head loads cleanly and reads
near chance on ten columns. Their table also publishes a per-label AUC profile on the 58 annotated
studies. Scoring those same studies and comparing the PROFILE — Effusion highest, MCL lowest — confirms or
refutes the transcription at no cost, and no number from this member should be believed before it has.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, override

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor, nn

from core.findings import TARGETS
from kneemri.inference.atlas_import import SOURCE, AtlasImport

logger = logging.getLogger(__name__)

CONFIG_NAME = "Qwen3-Embedding-8B-Atlas_small_d2d2d8_multiscale__pillar0_multimodal_ucsf_abd_ct_multiwindow.json"

CONFIG = SOURCE / "miniclip/model_configs" / CONFIG_NAME
"""The only config the publisher ships is for their CT modality. Every field of the trunk proper is shared
with the knee fine-tune — verified by an exact 747-tensor load with zero missing and zero unexpected keys —
so the MODALITY block is the only thing rebuilt from the checkpoint, and the rest is read from here rather
than retyped into this file as constants."""

MODALITY = "breast_mr"
"""The patch-embed the knee fine-tune actually carries. Pillar-0 is a BREAST MR foundation model; the
cross-anatomy transfer is the point of the candidate, not an oversight in this name."""

GRID = (192, 384, 384)
"""Depth, height, width the archive's adapter resamples every series to, before the channel replicate."""

LAYOUT = (GRID[0], GRID[1] // 2, GRID[2] // 2)
"""What the modality block's `image_size` has to say, which is NOT the volume that is fed.

`prepare_multiscale_layout` derives the window grid arithmetically as `image_size // patch_size` on all
three axes, but the patch embed's convolution stack reduces depth by 6 and the two in-plane axes by 12.
Declaring the true volume therefore builds a positional embedding for four times the tokens the trunk
actually produces, and the forward dies deep inside an attention block on a shape mismatch that names
neither this file nor the config field responsible. Halving the in-plane axes states the token grid the
convolutions really emit: measured, for every candidate volume tried, as exactly `build_scales`'s own
per-scale token counts."""

CHANNELS = 3
"""Their adapter replicates the single grey volume across three channels. Unlike `foreign_member`'s three,
this axis carries no depth: depth is its own dimension here."""

SCALE = 255.0

EPS = 1e-6
"""Keeps the log-odds round-trip in `forward` finite at a saturated probability."""

GOLD = 0.797
"""The macro-AUC the archive publishes for itself on the 58 radiologist-annotated studies.

It lives here rather than in the checkpoint because the checkpoint does not carry it — the archive states
it in prose. It is not a claim about whether this member is worth a seat: every lane ceiling this project
owns is above it. It is the ORACLE `pillar_gate` compares OUR feed against, which is the only thing that
can separate a low reading here from a fact about the model."""


PILLAR_LABELS: tuple[str, ...] = TARGETS
"""The head's twelve columns, transcribed from the archive README's per-label table in its printed order.
Identical to `TARGETS` here, which makes the permutation an identity — and makes it worth stating rather
than assuming, because a silent identity is indistinguishable from a permutation nobody checked."""


class AtlasArgs(TypedDict):
    """The one field of the publisher's `atlas_args` this file reads: how many scales the trunk pools."""

    num_scales: int


class AtlasModel(TypedDict):
    """The trunk half of the config, passed through to their constructor untouched apart from typing."""

    atlas_args: AtlasArgs


class AtlasConfig(TypedDict):
    """The publisher's `model_config` at the JSON boundary, typed rather than left as a free dict.

    A modality block is a stranger's schema and stays `dict[str, object]`: this file writes exactly two of
    its keys and forwards the rest, so naming the others would be transcription, not typing.
    """

    model: AtlasModel
    modalities: dict[str, dict[str, object]]


@dataclass(frozen=True)
class PillarSpec:
    """What the archive and the publisher's config together say about the function the weights hold."""

    dim: int
    labels: tuple[str, ...]
    model_config: AtlasConfig

    @staticmethod
    def of(payload: dict[str, dict[str, Tensor]]) -> "PillarSpec":
        """The spec this checkpoint describes, with the modality block rebuilt from its own tensors.

        `in_chans` is read off the patch-embed convolution rather than assumed, because that is the one
        field where the knee fine-tune could differ from the shipped CT config and still load a trunk.
        """
        shipped: AtlasConfig = json.loads(CONFIG.read_text(encoding="utf-8"))["vision_cfg"]["model_config"]
        block = dict(next(iter(shipped["modalities"].values())))
        embed = payload["model"][f"model.visual.patch_embeds.{MODALITY}.conv_down.0.weight"]
        block["image_size"] = list(LAYOUT)
        block["in_chans"] = int(embed.shape[1])
        return PillarSpec(
            dim=int(payload["head"]["weight"].shape[1]),
            labels=PILLAR_LABELS,
            model_config={"model": shipped["model"], "modalities": {MODALITY: block}},
        )

    @property
    def order(self) -> NDArray[np.int64]:
        """Rows of the head's output that put its findings in `TARGETS` order."""
        return np.array([self.labels.index(target) for target in TARGETS], dtype=np.int64)


class PillarMember(nn.Module):
    """The published 3D trunk and its linear head, answering for a whole study in `TARGETS` order."""

    def __init__(self, spec: PillarSpec) -> None:
        super().__init__()
        self.spec = spec
        self.trunk = AtlasImport.atlas_class()(
            args=None,
            embed_dim=spec.dim // int(spec.model_config["model"]["atlas_args"]["num_scales"]),
            num_classes=0,
            model_config=spec.model_config,
            modalities=spec.model_config["modalities"],
        )
        self.head = nn.Linear(spec.dim, len(TARGETS))
        self.register_buffer("permutation", torch.from_numpy(spec.order))

    def embedding(self, series: Float[Tensor, "c d h w"]) -> Float[Tensor, "1 dim"]:
        """One series' trunk features, L2-normalised — the state the head was fitted against.

        The trunk itself does not normalise: it max-pools each scale and concatenates, so what comes out
        has a positive mean and a norm around fifty. A linear head read against THAT produces logits in
        the hundreds, every sigmoid lands on exactly 0.0 or 1.0, and every AUC downstream is computed over
        ties — which is how three feeds differing in crop, depth AND resolution once returned bit-identical
        numbers on four findings. It is also self-refuting as a training state: a head fitted on unnormalised
        features would carry weights near zero, and these carry an absmax of 0.39, which is the magnitude a
        unit-norm input calls for. This is the one line separating a reading from an artefact.
        """
        return torch.nn.functional.normalize(self.trunk({MODALITY: series.unsqueeze(0)}).float(), dim=-1)

    @override
    def forward(self, volumes: Float[Tensor, "b k c d h w"]) -> Float[Tensor, "b n"]:
        """Log-odds of the MEAN SIGMOID over a study's series, which is the rule the archive publishes.

        The mean is taken in probability space and handed back as log-odds so that a reader applying its
        own sigmoid reproduces their rule exactly; averaging logits instead would be a different estimator
        wearing the same name. Series are run one at a time because one of them is already 28 million
        voxels, and a study's five would be the whole card.
        """
        chances = []
        for study in volumes:
            feats = (self.embedding(series) for series in study)
            per_series = [torch.sigmoid(self.head(features).float()) for features in feats]
            chances.append(torch.stack(per_series).mean(dim=0))
        stacked = torch.cat(chances).clamp(EPS, 1.0 - EPS)
        return torch.log(stacked / (1.0 - stacked)).index_select(1, self.get_buffer("permutation"))

    @staticmethod
    def rebuild(checkpoint: Path) -> "PillarMember":
        """The archive's function, weights and all, with its text tower dropped.

        The checkpoint holds a full CLIP pair; `model.text.*` is dead weight for a classifier and is not
        loaded. `strict=True` on what remains is the assertion that this rebuild is the published trunk
        and not something shaped enough like it to load.
        """
        payload: dict[str, dict[str, Tensor]] = torch.load(checkpoint, map_location="cpu", weights_only=True)
        member = PillarMember(PillarSpec.of(payload))
        prefix = "model.visual."
        member.trunk.load_state_dict(
            {name[len(prefix) :]: value for name, value in payload["model"].items() if name.startswith(prefix)}
        )
        member.head.load_state_dict(payload["head"])
        return member.eval()


@dataclass(frozen=True)
class VolumeFeed:
    """Our cached slot pixels, handed over as whole volumes rather than cut into per-slice windows.

    One slot is one series. A slot the decoder never filled is dropped rather than resampled from zeros,
    for the same reason `WindowFeed` drops it: an empty volume is not a missing volume to a mean over
    series, it is a confident prediction of nothing.
    """

    images: NDArray[np.uint8]
    mask: NDArray[np.float32]
    grid: tuple[int, int, int]

    @staticmethod
    def of(directory: Path, key: str) -> "VolumeFeed":
        """The cache named by `key`, memory-mapped, beside its per-slot presence mask."""
        images = np.load(directory / f"images_{key}.npy", mmap_mode="r")
        mask = np.load(directory / f"mask_{key}.npy")
        return VolumeFeed(images, mask.astype(np.float32), GRID)

    @staticmethod
    def order(directory: Path, key: str) -> list[str]:
        """The study order the cache's rows are indexed by."""
        return list(json.loads((directory / f"index_{key}.json").read_text(encoding="utf-8")))

    def volumes(self, row: int) -> UInt8[np.ndarray, "k d h w"]:
        """Every present slot of one study, each still at the cache's own depth and size."""
        study = np.asarray(self.images[row])
        present = [slot for slot in range(study.shape[0]) if self.mask[row, slot] > 0]
        return study[present] if present else study[:1]

    def batch(self, row: int, device: torch.device) -> Float[Tensor, "k c d h w"]:
        """One study's series, prepared exactly as the archive's adapter describes.

        Divide by 255, z-score PER SERIES, resample trilinearly to their grid, replicate to three
        channels — in that order. Normalising after the resample would z-score interpolated voxels, which
        is a different statistic on the same pixels.
        """
        pixels = torch.from_numpy(np.ascontiguousarray(self.volumes(row))).to(device).float() / SCALE
        flat = pixels.reshape(pixels.shape[0], -1)
        normalised = (pixels - flat.mean(dim=1).view(-1, 1, 1, 1)) / (flat.std(dim=1).view(-1, 1, 1, 1) + EPS)
        resampled = torch.nn.functional.interpolate(
            normalised.unsqueeze(1), size=self.grid, mode="trilinear", align_corners=False
        )
        return resampled.expand(-1, CHANNELS, -1, -1, -1)
