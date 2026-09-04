"""A published foreign checkpoint, rebuilt from its own archive and read through OUR corpus.

WHY IT EXISTS. `foreign_ruler` joined our members to six CC0 checkpoints on the 58 annotated studies and
found no member gap — the macro difference was +0.0016 against a standard error of ~0.012. That closes
the "their model is better" question and opens a different one: those six are genuinely FOREIGN members.
Different backbones (coatnet / convnextv2 / effnetv2 against our DINOv2), a different head (per-finding
attention over windows), a different corpus preprocessing and different labels. Sixteen scored board
entries built from OUR members span 0.936-0.941 at six to twelve seats, which is the null distribution
for reseating what we already have. A foreign family is the one axis that pool has never had.

Competition Rule 2.6a permits external data that is "publicly available and equally accessible to use by
all Participants ... at no cost"; both source datasets are CC0-1.0.

THE FEED IS THE RISK, AND IT IS GATED ELSEWHERE. Their pipeline fed bags of three physically-consecutive
slices from a corpus we do not have, at [0,1] with NO ImageNet normalisation. Ours are slot-routed,
cropped and cached at a different resolution. A foreign model run through our feed can read low because
the FEED is wrong, and that looks exactly like a fact about the model. `kneemri.evaluation.foreign_gate`
is the oracle that separates the two before any card hours are spent — this module serves, and never
imports the layer that scores what it serves.

WHAT IT DOES NOT DO. It does not fine-tune, and it does not adapt their weights to our pixels. It rebuilds
the published function exactly as its own checkpoint describes it — `arch`, `res` and the label order all
come from the archive rather than from this module's assumptions — and asks what that function says about
our studies. Anything else is a different experiment.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, override

import numpy as np
import timm
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor, nn

from core.findings import TARGETS
from kneemri.inference.foreign_head import ForeignHead

logger = logging.getLogger(__name__)

CHANNELS = 3
"""Slices per window. Their reader stacks three PHYSICALLY CONSECUTIVE slices into an RGB image, so the
three channels are a depth axis wearing a colour axis's clothes -- which is why no ImageNet mean is
subtracted anywhere below."""

SCALE = 255.0
"""Our cache stores uint8; their reader produced floats in [0,1]. This is the whole normalisation."""


class PublishedArchive(TypedDict):
    """The five fields a published archive is read for, at the JSON-ish boundary of a stranger's `.pt`.

    A TypedDict rather than a free dict because this is the only place the foreign layout is spelled, and
    the alternative is a permuted label list or a wrong read size reaching a model that loads cleanly.
    The archives carry more keys than these (`src`, `epoch`, `aucs`); nothing here reads them, and
    `foreign_ruler` owns `aucs`.
    """

    model: dict[str, Tensor]
    arch: str
    res: int
    lab: list[str]
    gold_auc: float


@dataclass(frozen=True)
class ForeignSpec:
    """What a published archive says about the function it holds — architecture, read size, label order.

    BESIDE ITS PRODUCER because every field is read off the checkpoint in one place. Rebuilding from
    assumptions instead is how a foreign member gets loaded at the wrong resolution or with its twelve
    outputs permuted, both of which load cleanly and score like a worse model.
    """

    arch: str
    res: int
    labels: tuple[str, ...]
    gold: float
    dim: int

    @staticmethod
    def of(payload: PublishedArchive) -> "ForeignSpec":
        """The spec a checkpoint's own fields describe, read in ONE place and nowhere else."""
        return ForeignSpec(
            arch=str(payload["arch"]),
            res=int(payload["res"]),
            labels=tuple(str(name) for name in payload["lab"]),
            gold=float(payload["gold_auc"]),
            dim=int(payload["model"]["clsW"].shape[1]),
        )

    @property
    def order(self) -> NDArray[np.int64]:
        """Rows of the head's output that put its findings in `TARGETS` order.

        Their label tuple is not ours and never was. A model whose twelve columns are permuted scores
        near chance on ten of them and plausibly on two, which reads as a bad model rather than a bug.
        """
        return np.array([self.labels.index(target) for target in TARGETS], dtype=np.int64)


class ForeignMember(nn.Module):
    """A published backbone and head, rebuilt from one archive and answering in `TARGETS` order."""

    def __init__(self, spec: ForeignSpec) -> None:
        super().__init__()
        self.spec = spec
        self.backbone = timm.create_model(
            spec.arch, pretrained=False, num_classes=0, in_chans=CHANNELS, global_pool="avg"
        )
        self.head = ForeignHead(spec.dim, len(TARGETS))
        self.register_buffer("permutation", torch.from_numpy(spec.order))

    @override
    def forward(self, windows: Float[Tensor, "b k c h w"]) -> Float[Tensor, "b n"]:
        """Logits for every finding, permuted into our order before anything downstream sees them."""
        batch, count = windows.shape[0], windows.shape[1]
        flat = windows.reshape(batch * count, *windows.shape[2:])
        features = self.backbone(flat).reshape(batch, count, -1)
        return self.head(features).index_select(1, self.permutation)

    @staticmethod
    def rebuild(checkpoint: Path) -> "ForeignMember":
        """The archive's function, weights and all, without executing its pickle.

        `weights_only=True` for the same reason `foreign_ruler` reads its AUCs that way: the argument for
        using a stranger's checkpoint at all is that nothing in it ever runs.
        """
        payload: PublishedArchive = torch.load(checkpoint, map_location="cpu", weights_only=True)
        member = ForeignMember(ForeignSpec.of(payload))
        prefix = "backbone."
        member.head.load_state_dict(ForeignHead.weights(payload["model"]))
        member.backbone.load_state_dict(
            {name[len(prefix) :]: value for name, value in payload["model"].items() if name.startswith(prefix)}
        )
        return member.eval()


@dataclass(frozen=True)
class WindowFeed:
    """Our cached slot pixels, cut into the bags of consecutive slices a foreign member was fitted on.

    Every window is three neighbouring slices of ONE slot: a bag that crossed a slot boundary would stack
    three unrelated planes into one RGB image, which is a picture no radiologist and no training corpus
    ever contained.
    """

    images: NDArray[np.uint8]
    mask: NDArray[np.float32]
    res: int

    @staticmethod
    def of(directory: Path, key: str, res: int) -> "WindowFeed":
        """The cache named by `key`, memory-mapped, beside its per-slot presence mask."""
        images = np.load(directory / f"images_{key}.npy", mmap_mode="r")
        mask = np.load(directory / f"mask_{key}.npy")
        return WindowFeed(images, mask.astype(np.float32), res)

    @staticmethod
    def order(directory: Path, key: str) -> list[str]:
        """The study order the cache's rows are indexed by."""
        return list(json.loads((directory / f"index_{key}.json").read_text(encoding="utf-8")))

    def windows(self, row: int) -> UInt8[np.ndarray, "k c h w"]:
        """Every three-slice window of every present slot, for one study.

        A slot the decoder never filled is dropped rather than fed as zeros: a black window is not a
        missing window to an attention pool, it is a window the pool can attend TO.
        """
        study = np.asarray(self.images[row])
        present = [slot for slot in range(study.shape[0]) if self.mask[row, slot] > 0]
        slices = study.shape[1]
        bags = [study[slot, start : start + CHANNELS] for slot in present for start in range(slices - CHANNELS + 1)]
        return np.stack(bags) if bags else study[:1, :CHANNELS]

    def batch(self, row: int, device: torch.device) -> Float[Tensor, "k c h w"]:
        """One study's windows at the member's own read size, in [0,1] and nothing else."""
        pixels = torch.from_numpy(np.ascontiguousarray(self.windows(row))).to(device).float() / SCALE
        if pixels.shape[-1] == self.res:
            return pixels
        return torch.nn.functional.interpolate(pixels, size=(self.res, self.res), mode="bilinear", align_corners=False)
