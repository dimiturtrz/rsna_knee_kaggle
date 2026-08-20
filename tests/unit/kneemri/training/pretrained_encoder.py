from pathlib import Path

import pytest
import torch
from torch import nn

from kneemri.training.pretrained_encoder import PretrainedEncoder

DIM = 8
SIDE = 12


class _Backbone(nn.Module):
    """A stand-in for `Dinov2Model`, written rather than downloaded: a unit test may not reach the network."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(SIDE * SIDE, DIM)


def test_variant(tmp_path: Path):
    """The file says which encoder it adapted, so a caller can refuse a mismatch."""
    path = tmp_path / "adapted.pt"
    torch.save({"encoder": {}, "variant": "small", "config": {}}, path)
    assert PretrainedEncoder(path).variant() == "small"


def test_into(tmp_path: Path):
    """A load that matches nothing must RAISE, not quietly leave the stock weights in place.

    This is the defect `encoder_drift` already met from the other side: a state dict whose keys the
    target does not name reported a drift of 0.000, which reads as "nothing changed" and meant "nothing
    was compared". Here the same silence would report the control's score under the arm's name.
    """
    backbone = _Backbone()
    weights = {name: torch.zeros_like(value) for name, value in backbone.state_dict().items()}
    path = tmp_path / "adapted.pt"
    torch.save({"encoder": weights, "variant": "small", "config": {}}, path)
    loaded = PretrainedEncoder(path).into(_Backbone(), "small")
    assert loaded.state_dict()["projection.weight"].abs().sum() == 0.0, "the adapted weights must land"

    with pytest.raises(ValueError, match="asked for base"):
        PretrainedEncoder(path).into(_Backbone(), "base")

    stranger = tmp_path / "stranger.pt"
    torch.save({"encoder": {"nothing.like.this": torch.zeros(1)}, "variant": "small", "config": {}}, stranger)
    with pytest.raises(ValueError, match="does not fit"):
        PretrainedEncoder(stranger).into(_Backbone(), "small")


def test_into_tolerates_the_mask_token(tmp_path: Path):
    """The one key a stock encoder legitimately lacks is the objective's own mask token.

    Its own case because the alternative to naming it is a blanket `strict=False`, which is the check
    turned off rather than a check that knows what it forgives.
    """
    backbone = _Backbone()
    backbone.mask_token = nn.Parameter(torch.zeros(1, 1, DIM))
    weights = {n: v for n, v in backbone.state_dict().items() if "mask_token" not in n}
    path = tmp_path / "adapted.pt"
    torch.save({"encoder": weights, "variant": "small", "config": {}}, path)
    assert PretrainedEncoder(path).into(backbone, "small") is backbone
