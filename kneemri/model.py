"""The study-level model: a shared image encoder under a per-diagnosis attention head.

Replicates the public baseline's architecture exactly, because the point of a replication is to
reproduce the number before changing anything — and because the trained members published alongside it
load only into this shape. `MemberPackage` checks that faithfulness numerically (see `members.py`);
this module is what it checks.
"""

from typing import override

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor, nn

from kneemri.slot_head import SlotHead

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

    def __init__(self, backbone: nn.Module, dim: int, n_slot: int, pool: str = "cls_mean") -> None:
        # `backbone` is any module callable as `backbone(pixel_values=x)` returning an object with
        # `last_hidden_state` of shape (n, 1 + patches, dim) — a DINOv2 from transformers in
        # production, a deterministic stand-in under test, and a medical-pretrained encoder when that
        # ablation runs. Kept structural rather than a named protocol so a substitution is a
        # constructor argument, not an edit here.
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.n_slot = n_slot
        self.head = SlotHead(dim * POOL_PARTS[pool], n_slot, len(TARGETS))
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
        batch, slots = imgs.shape[:2]
        x = imgs.reshape(batch * slots, *imgs.shape[2:]).float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            # The cache is held at the highest resolution any configuration needs and the rest
            # downsample from it, so every configuration sees the same pixels through a different
            # sampling grid rather than a different crop.
            x = nn.functional.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        tokens = self.backbone(pixel_values=x).last_hidden_state
        feat = self._pool(tokens).reshape(batch, slots, -1)
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
