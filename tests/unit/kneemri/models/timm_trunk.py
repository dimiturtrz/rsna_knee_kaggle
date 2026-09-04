"""The timm-keyed loader: the calling contract it wears, and where it looks for the file."""

from pathlib import Path

import torch
from torch import Tensor, nn

from kneemri.models.timm_trunk import TimmHubWeights, TimmMapWeights, TimmOptions, TimmTrunk

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


def test_resolve(tmp_path: Path):
    """A directory is joined with the stated filename; a file is taken as itself."""
    weights = TimmHubWeights("vit_base_patch16_224", DIM, filename="model.safetensors")
    given = tmp_path / "model.safetensors"
    given.write_bytes(b"")

    assert weights.resolve(tmp_path) == given
    assert weights.resolve(given) == given
