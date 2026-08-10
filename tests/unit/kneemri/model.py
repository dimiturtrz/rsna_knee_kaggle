from dataclasses import dataclass
from typing import override

import numpy as np
import pytest
import torch
from torch import Tensor

from kneemri.model import POOL_PARTS, TARGETS, StudyModel

SEED = 2026
DIM = 8
PATCHES = 16
SLOTS = 6
GROUP = 3


@dataclass
class _Output:
    last_hidden_state: Tensor


class _FakeBackbone(torch.nn.Module):
    """A stand-in encoder: deterministic tokens of the right shape, no download, no pretrained weights."""

    @override
    def forward(self, *, pixel_values: Tensor) -> _Output:
        n = pixel_values.shape[0]
        column = pixel_values.mean(dim=(1, 2, 3)).reshape(n, 1, 1)
        base = torch.arange(PATCHES + 1, dtype=torch.float32).reshape(1, -1, 1)
        return _Output(last_hidden_state=(base + column).expand(n, PATCHES + 1, DIM).contiguous())


@pytest.fixture
def study_model() -> StudyModel:
    torch.manual_seed(SEED)
    return StudyModel(_FakeBackbone(), DIM, SLOTS)


def test_forward(study_model: StudyModel):
    """One logit per diagnosis per study, and a masked-out slot must change the answer."""
    torch.manual_seed(SEED)
    imgs = torch.randint(0, 256, (2, SLOTS, GROUP, 14, 14), dtype=torch.uint8)
    partial = torch.ones(2, SLOTS)
    partial[:, -1] = 0.0
    study_model.eval()
    with torch.no_grad():
        every_slot = study_model.forward(imgs, torch.ones(2, SLOTS), 14)
        one_absent = study_model.forward(imgs, partial, 14)
    assert every_slot.shape == (2, len(TARGETS))
    assert not torch.allclose(every_slot, one_absent), "the presence mask must reach the softmax"


def test_forward_resizes_to_the_requested_size(study_model: StudyModel):
    """A cache held at one resolution is resampled to whatever the member was fitted at."""
    torch.manual_seed(SEED)
    imgs = torch.randint(0, 256, (1, SLOTS, GROUP, 28, 28), dtype=torch.uint8)
    study_model.eval()
    with torch.no_grad():
        out = study_model.forward(imgs, torch.ones(1, SLOTS), 14)
    assert out.shape == (1, len(TARGETS))


def test_fingerprint(study_model: StudyModel):
    """The same model answers the seeded question identically twice; a different model does not."""
    device = torch.device("cpu")
    first = study_model.fingerprint(14, GROUP, device)
    assert first.shape == (2, len(TARGETS))
    assert np.array_equal(first, study_model.fingerprint(14, GROUP, device))
    torch.manual_seed(SEED + 1)
    other = StudyModel(_FakeBackbone(), DIM, SLOTS)
    assert not np.allclose(first, other.fingerprint(14, GROUP, device))


def test_fingerprint_leaves_training_mode_untouched(study_model: StudyModel):
    """A check run mid-training must not silently leave the model in eval."""
    study_model.train()
    study_model.fingerprint(14, GROUP, torch.device("cpu"))
    assert study_model.training


def test_pool_parts_widths_match_the_head_input():
    """`cls_mean` concatenates two `dim`-wide parts, `cls_mean_focal` three — the head's width follows."""
    assert POOL_PARTS["cls_mean"] == 2
    torch.manual_seed(SEED)
    focal = StudyModel(_FakeBackbone(), DIM, 2, pool="cls_mean_focal")
    assert focal.head.proj[1].in_features == DIM * POOL_PARTS["cls_mean_focal"]
