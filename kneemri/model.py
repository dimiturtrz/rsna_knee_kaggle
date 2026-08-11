"""The study-level model: a shared image encoder under a per-diagnosis attention head.

Replicates the public baseline's architecture exactly, because the point of a replication is to
reproduce the number before changing anything — and because the trained members published alongside it
load only into this shape. `MemberPackage` checks that faithfulness numerically (see `members.py`);
this module is what it checks.
"""

from dataclasses import dataclass
from typing import override

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor, nn

from kneemri.anatomy import ANATOMY, SlotAffinity
from kneemri.slot_head import SlotHead
from kneemri.slots import SLOTS

TARGETS: tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
"""The twelve findings, in the column order the submission and the trained members both use."""

POOL_PARTS: dict[str, int] = {"cls_mean": 2, "cls_mean_focal": 3}
"""How many `dim`-wide parts each pooling scheme concatenates, which fixes the head's input width."""

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

FINGERPRINT_SEED = 2026
"""The seed of the synthetic bag. A constant of the question, not of the member that answers it."""

FINGERPRINT_BATCH = 2
"""Two studies, so the masked branch of the softmax is exercised alongside the unmasked one."""

_FOCAL_FRACTION = 8
"""The focal pooling keeps the top 1/8 of each channel's patch responses."""


@dataclass(frozen=True)
class StudyDesign:
    """The model's behavioural switches, as one value rather than as a growing argument list.

    They arrived one at a time — a pooling mode, then a slice aggregation, then the anatomical prior —
    and each was a string or a flag threaded through the constructor, the trainer, the checkpoint and
    every analysis that rebuilds a member. Grouping them means a new switch is one field rather than an
    edit at each of sixteen call sites, and it means a checkpoint can record the design it was trained
    under as one object.
    """

    pool: str = "cls_mean"
    slice_pooling: str = "channels"
    anatomical_prior: bool = False


class StudyModel(nn.Module):
    """Encoder plus head over a study's bag of slot images.

    The bag is flattened for the encoder and folded back before the head, so the encoder never sees the
    study structure and the head never sees pixels. Byte scaling, the ImageNet normalisation and the
    resize all live inside `forward` on purpose: they are part of the function the weights were fitted
    to, and a member run through a different one of them loads cleanly and predicts from the wrong
    image.
    """

    mean: Tensor
    std: Tensor

    def __init__(self, backbone: nn.Module, dim: int, n_slot: int, design: StudyDesign | None = None) -> None:
        # `backbone` is any module callable as `backbone(pixel_values=x)` returning an object with
        # `last_hidden_state` of shape (n, 1 + patches, dim) — a DINOv2 from transformers in
        # production, a deterministic stand-in under test, and a medical-pretrained encoder when that
        # ablation runs. Kept structural rather than a named protocol so a substitution is a
        # constructor argument, not an edit here.
        super().__init__()
        self.design = design or StudyDesign()
        self.backbone = backbone
        self.pool = self.design.pool
        self.slice_pooling = self.design.slice_pooling
        self.n_slot = n_slot
        # The prior is defined over the canonical six slots, so asking for it with a different slot
        # count is a mistake rather than something to silently truncate — a (12, 6) table broadcast
        # against a (12, 2) attention map would either throw deep inside the head or, worse, not.
        if self.design.anatomical_prior and n_slot != len(SLOTS):
            message = f"anatomical prior covers {len(SLOTS)} slots, model has {n_slot}"
            raise ValueError(message)
        prior = SlotAffinity(ANATOMY).matrix(TARGETS, SLOTS) if self.design.anatomical_prior else None
        self.head = SlotHead(dim * POOL_PARTS[self.pool], n_slot, len(TARGETS), prior)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @override
    def forward(
        self,
        imgs: UInt8[Tensor, "b s g p q"],
        mask: Float[Tensor, "b s"],
        img_size: int | None = None,
    ) -> Float[Tensor, "b o"]:
        """Encode every slot image of every study in the batch, then aggregate per diagnosis."""
        batch, slots, group = imgs.shape[:3]
        # "channels": three consecutive slices become the encoder's three channels — one forward per
        # slot. "max": each slice is encoded ALONE (replicated across the three channels the encoder
        # expects) and the slot's embedding is the per-dimension maximum over its slices.
        #
        # The difference is not a preference. Channel stacking couples the group's PHYSICAL THICKNESS
        # to the slice count, which is measurably why raising inference slices from 12 to 32 cost
        # -0.018 here while the recipe that pools per-slice can do it freely. It also means a finding
        # is only visible if it falls inside a sampled three-slice group.
        per_slice = self.slice_pooling == "max"
        if per_slice:
            x = imgs.reshape(batch * slots * group, 1, *imgs.shape[3:]).expand(-1, 3, -1, -1)
            x = x.float().div(255.0)
        else:
            x = imgs.reshape(batch * slots, *imgs.shape[2:]).float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            # The cache is held at the highest resolution any configuration needs and the rest
            # downsample from it, so every configuration sees the same pixels through a different
            # sampling grid rather than a different crop.
            x = nn.functional.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        tokens = self.backbone(pixel_values=x).last_hidden_state
        pooled = self._pool(tokens)
        feat = pooled.reshape(batch, slots, group, -1).amax(dim=2) if per_slice else pooled.reshape(batch, slots, -1)
        return self.head(feat, mask)

    def fingerprint(self, image_size: int, group: int, device: torch.device) -> NDArray[np.float32]:
        """This model's output on a fixed synthetic bag, as a portable identity.

        Weights read through the wrong preprocessing produce predictions, not errors: the submission is
        well formed, the log says nothing, and the difference is a number no output of the run reveals.
        So the input is generated from a seed rather than read — identical on any machine — and pushed
        through the whole forward path: byte scaling, ImageNet normalisation, resize, encoder, slot
        attention. Any of those differing moves the output by order one.

        float32 throughout: autocast would make the value depend on which device happened to run it,
        and the point of the number is that it does not.
        """
        generator = torch.Generator().manual_seed(FINGERPRINT_SEED)
        imgs = torch.randint(
            0,
            256,
            (FINGERPRINT_BATCH, self.n_slot, group, image_size, image_size),
            generator=generator,
            dtype=torch.uint8,
        ).to(device)
        mask = torch.ones(FINGERPRINT_BATCH, self.n_slot, device=device)
        mask[1, -1] = 0.0
        was_training = self.training
        self.eval()
        with torch.no_grad():
            out = self(imgs, mask, image_size).float().cpu().numpy()
        if was_training:
            self.train()
        return out

    def load_trained(self, state: dict[str, Tensor]) -> None:
        """Load weights saved by any earlier version of this model.

        Every checkpoint written before the anatomical prior existed lacks `head.prior`, and a strict
        load rejects the lot — including the five members already on the leaderboard and the ones the
        Kaggle kernel mounts. A missing prior is not ambiguous: the parameter is zeros when the design
        does not use it, which is exactly the model this checkpoint was trained as.

        So keys the checkpoint does not carry keep their initialised value, and everything it does carry
        must still match in name and shape. That is deliberately narrower than `strict=False`, which
        would also swallow a checkpoint that had nothing to do with this model.
        """
        merged = self.state_dict() | state
        self.load_state_dict(merged)

    def parameter_groups(self, head_lr: float, backbone_lr: float, unfreeze_last: int) -> list[dict[str, object]]:
        """Freeze the encoder except its last blocks, and give what remains its own learning rate.

        The model owns this rather than the trainer: which blocks exist, and therefore which of them can
        be opened, is a fact about the encoder, not about the fitting procedure.
        """
        blocks = list(self.backbone.encoder.layer)  # pyrefly: ignore[missing-attribute]
        trainable = {id(parameter) for block in blocks[-unfreeze_last:] for parameter in block.parameters()}
        opened: list[torch.nn.Parameter] = []
        frozen: list[torch.nn.Parameter] = []
        for parameter in self.backbone.parameters():
            (opened if id(parameter) in trainable else frozen).append(parameter)
        for parameter in frozen:
            # torch takes this positionally; ruff's boolean-trap rule and the C signature disagree.
            parameter.requires_grad_(False)  # noqa: FBT003
        return [
            {"params": list(self.head.parameters()), "lr": head_lr},
            {"params": opened, "lr": backbone_lr},
        ]

    def _pool(self, tokens: Float[Tensor, "n t d"]) -> Float[Tensor, "n f"]:
        """Reduce patch tokens to one vector: the class token, the patch mean, optionally the tail."""
        patch = tokens[:, 1:]
        parts = [tokens[:, 0], patch.mean(1)]
        if self.pool == "cls_mean_focal":
            # The upper tail of each channel over the patch grid, taken per channel rather than by
            # selecting whole patches: a finding occupies a small part of the field, so a plain mean
            # over the patches dilutes it, and this keeps the top eighth of each channel's responses.
            k = max(1, patch.shape[1] // _FOCAL_FRACTION)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        return torch.cat(parts, dim=1)
