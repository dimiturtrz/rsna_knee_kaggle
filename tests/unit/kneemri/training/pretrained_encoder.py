from pathlib import Path

import pytest
import torch
from torch import nn

from kneemri.training.pretrained_encoder import ALL_BLOCKS, UNSTAMPED, PretrainedEncoder

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


def test_cache(tmp_path: Path):
    """The slice population an adaptation SAW, and an honest answer when the file predates the stamp.

    A file with no key must not read as one whose cache matches: both adaptations on disk were run on
    the default 12-slice cache while every arm importing them trains on 20 slices at 0.10-0.90, and the
    only record of that was a line in a launch log.
    """
    stamped = tmp_path / "stamped.pt"
    torch.save({"encoder": {}, "variant": "small", "cache": "336px_20sl_0.1-0.9band", "config": {}}, stamped)
    assert PretrainedEncoder(stamped).cache() == "336px_20sl_0.1-0.9band"

    older = tmp_path / "older.pt"
    torch.save({"encoder": {}, "variant": "small", "config": {}}, older)
    assert PretrainedEncoder(older).cache() == UNSTAMPED


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


class _Blocks(nn.Module):
    """A stand-in shaped like the encoder's repeated blocks, so a depth actually has something to cut."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Linear(SIDE, DIM)
        self.encoder = nn.ModuleDict({"layer": nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(3)])})
        self.layernorm = nn.LayerNorm(DIM)


def _adapted(into: Path, weights: dict[str, torch.Tensor]) -> Path:
    path = into / "adapted.pt"
    torch.save({"encoder": weights, "variant": "small", "config": {}}, path)
    return path


def test_upto(tmp_path: Path):
    """A depth takes the patch embedding and the blocks below it, and leaves the final norm stock.

    The norm is calibrated to the activations of the last block; importing it over stock blocks is a
    mismatch neither side asked for.
    """
    model = _Blocks()
    kept = PretrainedEncoder(_adapted(tmp_path, dict(model.state_dict()))).upto(2)

    assert "embeddings.weight" in kept
    assert "encoder.layer.0.weight" in kept
    assert "encoder.layer.1.weight" in kept
    assert "encoder.layer.2.weight" not in kept, "a block at or past the depth stays stock"
    assert "layernorm.weight" not in kept, "the final norm belongs to the block it follows"


def test_upto_takes_everything_by_default(tmp_path: Path):
    model = _Blocks()
    saved = dict(model.state_dict())

    assert set(PretrainedEncoder(_adapted(tmp_path, saved)).upto(ALL_BLOCKS)) == set(saved)


def test_into_at_a_depth_leaves_the_rest_stock(tmp_path: Path):
    """The withheld keys are missing BY REQUEST, so the loader must not read them as a bad fit."""
    adapted = _Blocks()
    zeroed = {name: torch.zeros_like(value) for name, value in adapted.state_dict().items()}
    target = _Blocks()
    stock_top = target.state_dict()["encoder.layer.2.weight"].clone()

    loaded = PretrainedEncoder(_adapted(tmp_path, zeroed)).into(target, "small", 2)

    state = loaded.state_dict()
    assert state["encoder.layer.0.weight"].abs().sum() == 0.0, "the blocks below the depth must land"
    assert torch.equal(state["encoder.layer.2.weight"], stock_top), "the blocks above it must not"


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
