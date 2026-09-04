"""The timm-keyed loader: the calling contract it wears, and where it looks for the file."""

from pathlib import Path

import torch
from torch import Tensor, nn

from kneemri.models.timm_trunk import TimmHubWeights, TimmTrunk

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


def test_resolve(tmp_path: Path):
    """A directory is joined with the stated filename; a file is taken as itself."""
    weights = TimmHubWeights("vit_base_patch16_224", DIM, filename="model.safetensors")
    given = tmp_path / "model.safetensors"
    given.write_bytes(b"")

    assert weights.resolve(tmp_path) == given
    assert weights.resolve(given) == given
