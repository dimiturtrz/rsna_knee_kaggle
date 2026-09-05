"""The timm-keyed loader: the calling contract it wears, and where it looks for the file."""

from pathlib import Path

import torch
from torch import Tensor, nn

from kneemri.models.timm_trunk import (
    TimmChunkedWeights,
    TimmHubWeights,
    TimmMapWeights,
    TimmOptions,
    TimmTrunk,
)

DIM = 8
PATCHES = 5


class _FakeTimm(nn.Module):
    """A timm ViT's half of the contract: `forward_features`, and a checkpointing switch spelt timm's way."""

    def __init__(self) -> None:
        super().__init__()
        self.grad_checkpointing = False

    def set_grad_checkpointing(self, *, enable: bool) -> None:
        self.grad_checkpointing = enable

    def forward_features(self, pixel_values: Tensor) -> Tensor:
        return torch.zeros(pixel_values.shape[0], 1 + PATCHES, DIM)


def test_forward():
    """The sequence arrives UNCHANGED — one prefix, not two, which is the whole reason for the class."""
    trunk = TimmTrunk(_FakeTimm(), DIM)

    out = trunk.forward(torch.zeros(2, 3, 16, 16))

    assert out.last_hidden_state.shape == (2, 1 + PATCHES, DIM)
    assert trunk.config.hidden_size == DIM


def test_is_gradient_checkpointing():
    """Read off the trunk that does the checkpointing, so the readback proves the flag LANDED."""
    trunk = TimmTrunk(_FakeTimm(), DIM)

    assert trunk.is_gradient_checkpointing is False


def test_gradient_checkpointing_enable():
    """Transformers' hook, with transformers' kwargs, reaching timm's differently-named switch."""
    trunk = TimmTrunk(_FakeTimm(), DIM)

    trunk.gradient_checkpointing_enable({"use_reentrant": False})

    assert trunk.is_gradient_checkpointing is True


def test_timm_hub_weights_options():
    """A ViT asks for `dynamic_img_size`, which is what lets a 224-grid checkpoint read 336."""
    expected: TimmOptions = {"dynamic_img_size": True}

    assert TimmHubWeights("vit_base_patch16_224", DIM).options() == expected


def test_timm_map_weights_options():
    """A hierarchical trunk asks for NOTHING — timm refuses the keyword on a model with no table to
    interpolate, so the override is the difference between the two families rather than a default."""
    assert TimmMapWeights("coatnet_2_rw_224", DIM).options() == {}


def test_timm_hub_weights_resolve(tmp_path: Path):
    """A directory is joined with the stated filename; a file is taken as itself."""
    weights = TimmHubWeights("vit_base_patch16_224", DIM, filename="model.safetensors")
    given = tmp_path / "model.safetensors"
    given.write_bytes(b"")

    assert weights.resolve(tmp_path) == given
    assert weights.resolve(given) == given


def test_timm_chunked_weights_resolve(tmp_path: Path):
    """A mount is still a directory or a file, and the local root is reached only when neither is given."""
    weights = TimmChunkedWeights("vit_base_patch16_224", DIM, filename="w.pth", directory="mricore")
    given = tmp_path / "w.pth"
    given.write_bytes(b"")

    assert weights.resolve(tmp_path) == given
    assert weights.resolve(given) == given


def test_flattened():
    """Chunked block keys flatten on their SECOND index, the prefix goes, and the SSL keys are left behind.

    The second index is the global block number because each DINOv2 chunk is padded at the front with
    identities — so `blocks.3.9` is block nine, and a reader who multiplied the chunk out would put it at
    eighteen. Asserted on a checkpoint whose chunk boundaries are the real ones (0,0) and (3,9).
    """
    weights = TimmChunkedWeights(
        "vit_base_patch16_224", DIM, section="teacher", prefix="backbone.", dropped=("mask_token",)
    )
    stored = {
        "teacher": {
            "backbone.blocks.0.0.norm1.weight": torch.zeros(1),
            "backbone.blocks.3.9.mlp.fc1.bias": torch.zeros(2),
            "backbone.cls_token": torch.zeros(3),
            "backbone.mask_token": torch.zeros(4),
            "dino_head.last_layer.weight_v": torch.zeros(5),
        }
    }

    flattened = weights.flattened(stored)

    assert sorted(flattened) == ["blocks.0.norm1.weight", "blocks.9.mlp.fc1.bias", "cls_token"]
