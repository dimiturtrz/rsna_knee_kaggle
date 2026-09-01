"""A trunk rebuilt from a stated architecture plus a bare state dict, and the contract it then wears."""

from pathlib import Path

import torch
from torch import nn

from kneemri.models.state_dict_trunk import (
    RADIMAGENET_DENSENET121,
    FeatureMapOutput,
    StateDictTrunk,
    StateDictWeights,
)

PREFIX = "backbone.0."
WIDTH = 5


def tiny() -> nn.Module:
    """A two-layer stand-in for a released trunk, so the test asserts the mechanism and not torchvision."""
    return nn.Sequential(nn.Conv2d(3, WIDTH, kernel_size=3, padding=1), nn.BatchNorm2d(WIDTH))


class Wrapped(nn.Module):
    """A model whose trunk is a named attribute, which is the shape `trunk_of` exists to reach into."""

    def __init__(self) -> None:
        super().__init__()
        self.features = tiny()


def written(directory: Path, *, prefix: str = PREFIX) -> StateDictWeights:
    """A checkpoint on disk under `prefix`, plus the declaration that rebuilds it."""
    stored = {prefix + key: value for key, value in tiny().state_dict().items()}
    torch.save(stored, directory / "trunk.pt")
    return StateDictWeights(
        architecture=tiny,
        trunk_of=lambda model: model,
        prefix=prefix,
        filename="trunk.pt",
    )


def test_resolve(tmp_path: Path) -> None:
    """A directory resolves to the declared filename inside it, and a file resolves to itself."""
    weights = written(tmp_path)
    assert weights.resolve(tmp_path) == tmp_path / "trunk.pt"
    assert weights.resolve(tmp_path / "trunk.pt") == tmp_path / "trunk.pt"


def test_build(tmp_path: Path) -> None:
    """The prefix is stripped and the weights land in the stated architecture, strictly."""
    built = written(tmp_path).build(tmp_path)
    assert isinstance(built, StateDictTrunk)
    emitted = built(pixel_values=torch.zeros(1, 3, 8, 8))
    assert emitted.last_hidden_state.shape == (1, WIDTH, 8, 8)


def test_build_refuses_a_wrong_architecture(tmp_path: Path) -> None:
    """A stated architecture the weights did not come off is refused, not silently left initialised."""
    written(tmp_path)
    mismatched = StateDictWeights(
        architecture=lambda: nn.Sequential(nn.Conv2d(3, WIDTH + 1, kernel_size=3, padding=1)),
        trunk_of=lambda model: model,
        prefix=PREFIX,
        filename="trunk.pt",
    )
    try:
        mismatched.build(tmp_path)
    except RuntimeError:
        return
    raise AssertionError("a state dict whose keys do not match the architecture must not load")


def test_forward() -> None:
    """The wrapper answers `pixel_values=` and returns `.last_hidden_state`, which the adapter reads."""
    trunk = StateDictTrunk(tiny())
    emitted = trunk.forward(pixel_values=torch.zeros(2, 3, 4, 4))
    assert isinstance(emitted, FeatureMapOutput)
    assert emitted.last_hidden_state.shape == (2, WIDTH, 4, 4)
    assert trunk(pixel_values=torch.zeros(2, 3, 4, 4)).last_hidden_state.shape == emitted.last_hidden_state.shape


def test_densenet_features() -> None:
    """It reaches the convolutional stack, which is the only part a released DenseNet checkpoint covers."""
    model = Wrapped()
    assert StateDictTrunk.densenet_features(model) is model.features


def test_radimagenet_densenet121_declaration() -> None:
    """The declaration names the prefix its 725 keys carry and the file the mirror publishes."""
    assert RADIMAGENET_DENSENET121.prefix == PREFIX
    assert RADIMAGENET_DENSENET121.filename == "DenseNet121.pt"
    assert RADIMAGENET_DENSENET121.repository == "Lab-Rasool/RadImageNet"
