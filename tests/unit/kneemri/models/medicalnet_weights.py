"""The mirror for the checkpoint loader — the one half that can be checked without their weights."""

import torch

from kneemri.models.medicalnet_weights import PARALLEL_PREFIX, MedicalNetWeights


def test_stripped() -> None:
    """The `DataParallel` prefix comes off every parameter it is on."""
    state = {f"{PARALLEL_PREFIX}layer4.2.bn3.weight": torch.zeros(1)}
    assert list(MedicalNetWeights.stripped(state)) == ["layer4.2.bn3.weight"]


def test_stripped_is_empty_for_an_unwrapped_checkpoint() -> None:
    """An empty result means "not wrapped", which is why the caller falls back to the whole state."""
    assert MedicalNetWeights.stripped({"layer4.2.bn3.weight": torch.zeros(1)}) == {}
