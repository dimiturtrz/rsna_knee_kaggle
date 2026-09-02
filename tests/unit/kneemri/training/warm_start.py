"""The mirror for `WarmStart`: every public method called, and asserted."""

from pathlib import Path

import pytest
import torch
from torch import nn

from kneemri.training.warm_start import STATE, WarmStart


def _saved(tmp_path: Path, module: nn.Module, epoch: int = 4) -> Path:
    destination = tmp_path / "member.pt"
    torch.save({STATE: module.state_dict(), "epoch": epoch}, destination)
    return destination


def test_applied(tmp_path: Path) -> None:
    """The loaded model computes what the SAVED one computed, and says where it came from."""
    source = nn.Linear(3, 2)
    checkpoint = _saved(tmp_path, source)
    target = nn.Linear(3, 2)
    line = WarmStart(checkpoint).applied(target)
    batch = torch.ones(1, 3)
    assert torch.allclose(target(batch), source(batch))
    assert "member.pt" in line
    assert "epoch 4" in line


def test_applied_refuses_a_payload_that_is_not_a_member(tmp_path: Path) -> None:
    """A bare state dict is the near miss, and loading it leniently would train something else."""
    destination = tmp_path / "bare.pt"
    torch.save(nn.Linear(3, 2).state_dict(), destination)
    with pytest.raises(ValueError, match="not a member checkpoint"):
        WarmStart(destination).applied(nn.Linear(3, 2))


def test_applied_refuses_a_different_architecture(tmp_path: Path) -> None:
    """Strict by design: a checkpoint that half-fits is the failure this class exists to make loud."""
    checkpoint = _saved(tmp_path, nn.Linear(3, 2))
    with pytest.raises(RuntimeError):
        WarmStart(checkpoint).applied(nn.Linear(5, 2))
