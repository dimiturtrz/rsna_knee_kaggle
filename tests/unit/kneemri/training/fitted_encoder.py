"""Converting a fitted member into the encoder file `--encoder-weights` reads."""

from pathlib import Path

import torch
from torch import Tensor, nn

from kneemri.models.lora import LoRALinear
from kneemri.training.fitted_encoder import FittedEncoder
from kneemri.training.pretrained_encoder import PretrainedEncoder

VARIANT = "facebook/dinov2-base"
CACHE = "336px_20sl"
HOLDOUT = 0.8412
RANK = 4
IN_FEATURES = 6
OUT_FEATURES = 5
SEED = 2026


def _adapted() -> LoRALinear:
    """A LoRA layer whose correction is NOT zero — a freshly built one folds to the base exactly."""
    torch.manual_seed(SEED)
    adapted = LoRALinear(nn.Linear(IN_FEATURES, OUT_FEATURES), RANK)
    with torch.no_grad():
        adapted.up.weight.copy_(torch.randn_like(adapted.up.weight))
    return adapted


def _member(root: Path, state: dict[str, Tensor]) -> Path:
    """A member checkpoint carrying `state` under the two prefixes a real one carries."""
    written = root / "member.pt"
    torch.save(
        {
            "model": {f"backbone.{name}": value for name, value in state.items()},
            "holdout": HOLDOUT,
            "epoch": 7,
            "config": {"variant": VARIANT, "cache_studies": CACHE},
        },
        written,
    )
    return written


def test_variant(tmp_path: Path):
    assert FittedEncoder(_member(tmp_path, {"w": torch.zeros(1)})).variant() == VARIANT


def test_cache(tmp_path: Path):
    assert FittedEncoder(_member(tmp_path, {"w": torch.zeros(1)})).cache() == CACHE


def test_holdout(tmp_path: Path):
    assert FittedEncoder(_member(tmp_path, {"w": torch.zeros(1)})).holdout() == HOLDOUT


def test_backbone_state(tmp_path: Path):
    """The `backbone.` prefix comes off — a stock encoder has never heard of it."""
    fitted = FittedEncoder(_member(tmp_path, {"embeddings.weight": torch.ones(2)}))

    assert list(fitted.backbone_state()) == ["embeddings.weight"]


def test_backbone_state_drops_a_compiled_prefix(tmp_path: Path):
    """`_orig_mod.` once made a drift measurement compare every key against a name that does not exist."""
    written = tmp_path / "compiled.pt"
    torch.save(
        {
            "model": {"_orig_mod.backbone.embeddings.weight": torch.ones(2)},
            "holdout": HOLDOUT,
            "epoch": 1,
            "config": {"variant": VARIANT, "cache_studies": CACHE},
        },
        written,
    )

    assert list(FittedEncoder(written).backbone_state()) == ["embeddings.weight"]


def test_backbone_state_keeps_nothing_outside_the_backbone(tmp_path: Path):
    """A head is fitted for one recipe's targets; importing it into another run is not a warm start."""
    written = tmp_path / "whole.pt"
    torch.save(
        {
            "model": {"backbone.embeddings.weight": torch.ones(2), "head.prior": torch.zeros(3)},
            "holdout": HOLDOUT,
            "epoch": 1,
            "config": {"variant": VARIANT, "cache_studies": CACHE},
        },
        written,
    )

    assert list(FittedEncoder(written).backbone_state()) == ["embeddings.weight"]


def test_merge_adapters():
    """The merged weight must be what the adapted layer actually computes, not the base it wraps."""
    adapted = _adapted()
    state = {f"query.{name}": value for name, value in adapted.state_dict().items()}

    merged = FittedEncoder.merge_adapters(state)

    assert torch.allclose(merged["query.weight"], adapted.merged().weight)
    assert torch.allclose(merged["query.bias"], adapted.merged().bias)


def test_merge_adapters_drops_the_adapter_keys():
    """An adapter key left in place is a key the stock architecture refuses to load."""
    state = {f"query.{name}": value for name, value in _adapted().state_dict().items()}

    assert sorted(FittedEncoder.merge_adapters(state)) == ["query.bias", "query.weight"]


def test_merge_adapters_leaves_an_unadapted_tensor_alone():
    tensor = torch.arange(3, dtype=torch.float32)

    assert torch.equal(FittedEncoder.merge_adapters({"embeddings.weight": tensor})["embeddings.weight"], tensor)


def test_correction():
    """`up @ down * alpha / rank` — the scale `LoRALinear.forward` applies, read off the shapes."""
    adapted = _adapted()
    state = {f"query.{name}": value for name, value in adapted.state_dict().items()}

    expected = (adapted.up.weight @ adapted.down.weight) * adapted.scale

    assert torch.allclose(FittedEncoder.correction(state, "query", torch.float32), expected)


def test_write(tmp_path: Path):
    """The written file is one `PretrainedEncoder` reads, which is the whole point of converting."""
    source = _member(tmp_path, {"embeddings.weight": torch.ones(2)})
    destination = tmp_path / "encoder.pt"

    FittedEncoder(source).write(destination)
    read = PretrainedEncoder(destination)

    assert read.variant() == VARIANT
    assert read.cache() == CACHE
    assert torch.equal(read.upto(0)["embeddings.weight"], torch.ones(2))
