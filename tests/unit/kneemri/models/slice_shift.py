from copy import deepcopy

import torch
from torch import nn

from kneemri.models.slice_shift import SliceShift


def _shift(blocks: int = 1, dim: int = 3) -> SliceShift:
    return SliceShift([nn.Identity() for _ in range(blocks)], dim)


def test_mixed_is_the_identity_on_a_single_slice_group():
    shift = _shift()
    tokens = torch.arange(24.0).view(4, 2, 3)

    assert torch.equal(shift.mixed(tokens, torch.ones(2, 3)), tokens)


def test_mixed_is_the_identity_at_zero_gates():
    shift = _shift()
    shift.group = 3
    tokens = torch.arange(36.0).view(6, 2, 3)

    assert torch.equal(shift.mixed(tokens, torch.zeros(2, 3)), tokens)


def test_mixed():
    """Gated neighbours are added, and nothing crosses from one stack into the next."""
    shift = _shift(dim=1)
    shift.group = 3
    tokens = torch.tensor([1.0, 2.0, 4.0, 10.0, 20.0, 40.0]).view(6, 1, 1)
    gate = torch.tensor([[0.5], [0.25]])

    mixed = shift.mixed(tokens, gate).flatten().tolist()

    assert mixed == [1.5, 3.5, 5.0, 15.0, 35.0, 50.0]


def test_hook_mixes_the_block_input_through_its_own_gate():
    block = nn.Identity()
    shift = SliceShift([block], 1)
    shift.group = 2
    with torch.no_grad():
        shift.gates[0] = torch.tensor([[1.0], [0.0]])

    out = block(torch.tensor([1.0, 2.0]).view(2, 1, 1))

    assert out.flatten().tolist() == [1.0, 3.0]


def test_a_deep_copy_hooks_its_own_gates_not_the_originals():
    model = nn.Sequential(nn.Identity())
    model.shift = SliceShift([model[0]], 1)
    model.shift.group = 2
    copy = deepcopy(model)
    with torch.no_grad():
        copy.shift.gates[0] = torch.tensor([[1.0], [0.0]])

    out = copy[0](torch.tensor([1.0, 2.0]).view(2, 1, 1))

    assert out.flatten().tolist() == [1.0, 3.0]
