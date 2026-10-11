"""The mirror for the checkpoint loader — the one half that can be checked without their weights."""

from pathlib import Path

import pytest
import torch

from kneemri.models.medicalnet import CHECKPOINT, DEPTH, READ_SIZE, MedicalNetTrunk
from kneemri.models.medicalnet_weights import MODULE, PARALLEL_PREFIX, STATE, MedicalNetWeights


def test_stripped() -> None:
    """The `DataParallel` prefix comes off every parameter it is on."""
    state = {f"{PARALLEL_PREFIX}layer4.2.bn3.weight": torch.zeros(1)}
    assert list(MedicalNetWeights.stripped(state)) == ["layer4.2.bn3.weight"]


def test_stripped_is_empty_for_an_unwrapped_checkpoint() -> None:
    """An empty result means "not wrapped", which is why the caller falls back to the whole state."""
    assert MedicalNetWeights.stripped({"layer4.2.bn3.weight": torch.zeros(1)}) == {}


THEIRS = """\
from torch import nn


class ResNet(nn.Module):
    def __init__(self, **geometry):
        super().__init__()
        self.geometry = geometry
        self.conv1 = nn.Linear(2, 2)
        self.conv_seg = nn.Linear(2, 1)


def resnet50(**geometry):
    return ResNet(**geometry)
"""
"""A stand-in for their `models/resnet.py`: one trunk layer and the segmentation head they never saved."""


def _weights(tmp_path: Path, state: dict[str, torch.Tensor]) -> MedicalNetWeights:
    checkout = tmp_path / "MedicalNet"
    (checkout / "models").mkdir(parents=True)
    (checkout / MODULE).write_text(THEIRS, encoding="utf-8")
    directory = tmp_path / "pretrain"
    directory.mkdir()
    torch.save({STATE: state}, directory / CHECKPOINT)
    return MedicalNetWeights(directory, checkout)


def _trunk_only() -> dict[str, torch.Tensor]:
    return {"conv1.weight": torch.ones(2, 2), "conv1.bias": torch.zeros(2)}


def test_network_is_their_constructor_at_this_probe_geometry(tmp_path: Path) -> None:
    network = _weights(tmp_path, _trunk_only()).network()
    assert vars(network)["geometry"]["sample_input_D"] == DEPTH
    assert vars(network)["geometry"]["sample_input_H"] == READ_SIZE


def test_trunk_loads_a_wrapped_checkpoint_frozen_with_only_the_head_missing(tmp_path: Path) -> None:
    wrapped = {f"{PARALLEL_PREFIX}{key}": value for key, value in _trunk_only().items()}
    trunk = _weights(tmp_path, wrapped).trunk(torch.device("cpu"))
    assert isinstance(trunk, MedicalNetTrunk)
    assert torch.equal(trunk.resnet.state_dict()["conv1.weight"], torch.ones(2, 2)), "loaded, not the init"
    assert not any(parameter.requires_grad for parameter in trunk.parameters())
    assert not trunk.resnet.training


def test_trunk_loads_an_unwrapped_checkpoint_whole(tmp_path: Path) -> None:
    trunk = _weights(tmp_path, _trunk_only()).trunk(torch.device("cpu"))
    assert torch.equal(trunk.resnet.state_dict()["conv1.weight"], torch.ones(2, 2))


def test_trunk_refuses_a_checkpoint_missing_more_than_the_head(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not fit"):
        _weights(tmp_path, {"conv1.weight": torch.ones(2, 2)}).trunk(torch.device("cpu"))


def test_trunk_refuses_a_key_the_network_does_not_have(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unexpected=\\['layer9"):
        _weights(tmp_path, {**_trunk_only(), "layer9.weight": torch.zeros(1)}).trunk(torch.device("cpu"))
