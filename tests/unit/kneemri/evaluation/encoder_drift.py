"""The drift gate's arithmetic, on hand-built state dicts where the right answer is known by hand."""

from pathlib import Path

import pytest
import torch

from kneemri.evaluation.encoder_drift import STEM, EncoderDrift
from kneemri.models.variants import SMALL


def _checkpoint(into: Path, *, compiled: bool = False, variant: str = SMALL, unfreeze_last: int = 1) -> Path:
    """A checkpoint shaped like a member's: a `model` state dict and the config the trainer records."""
    prefix = "backbone._orig_mod." if compiled else "backbone."
    path = into / "member.pt"
    torch.save(
        {
            "config": {"variant": variant, "unfreeze_last": unfreeze_last, "unfreeze_offset": 0},
            "model": {
                f"{prefix}embeddings.cls_token": torch.ones(4),
                f"{prefix}encoder.layer.0.weight": torch.ones(4),
                f"{prefix}encoder.layer.1.weight": torch.full((4,), 3.0),
                "head.classifier.weight": torch.zeros(4),
            },
        },
        path,
    )
    return path


def _stock() -> dict[str, torch.Tensor]:
    """The pretrained side: block 0 unmoved, block 1 moved by exactly half its own norm."""
    return {
        "embeddings.cls_token": torch.ones(4),
        "encoder.layer.0.weight": torch.ones(4),
        "encoder.layer.1.weight": torch.full((4,), 2.0),
    }


def _adapted(into: Path) -> Path:
    """A checkpoint shaped like `kneemri.training.pretrain` writes: the encoder alone, stock-keyed."""
    path = into / "adapted.pt"
    torch.save({"variant": SMALL, "config": {}, "encoder": dict(_stock())}, path)
    return path


def test_variant(tmp_path: Path) -> None:
    assert EncoderDrift(_checkpoint(tmp_path)).variant() == SMALL
    assert EncoderDrift(_adapted(tmp_path)).variant() == SMALL, "an adaptation records it at the top level"


def test_source(tmp_path: Path) -> None:
    """Both shapes reach the same arithmetic: a member behind `backbone.`, an adaptation behind nothing."""
    member, prefix = EncoderDrift(_checkpoint(tmp_path)).source()
    assert prefix == "backbone."
    assert "backbone.encoder.layer.0.weight" in member

    adapted, no_prefix = EncoderDrift(_adapted(tmp_path)).source()
    assert no_prefix == ""
    assert set(adapted) == set(_stock())


def test_blocks_of_an_adaptation(tmp_path: Path) -> None:
    """An SSL checkpoint measures against the same stock weights, with no prefix to strip."""
    assert EncoderDrift(_adapted(tmp_path)).blocks(_stock()) == {STEM: 0.0, 0: 0.0, 1: 0.0}


def test_repository(tmp_path: Path) -> None:
    assert EncoderDrift(_checkpoint(tmp_path)).repository() == "facebook/dinov2-small"


def test_trained(tmp_path: Path) -> None:
    """The head is dropped and both prefixes are stripped, so the keys match the stock model's."""
    trained = EncoderDrift(_checkpoint(tmp_path, compiled=True)).trained()

    assert set(trained) == set(_stock())


def test_grouped(tmp_path: Path) -> None:
    """Block index comes from the first integer segment; everything else lands under STEM."""
    drift = EncoderDrift(_checkpoint(tmp_path))

    groups = drift.grouped(drift.trained())

    assert sorted(groups) == [STEM, 0, 1]
    assert groups[STEM] == ["embeddings.cls_token"]


def test_blocks(tmp_path: Path) -> None:
    """An unmoved block reads exactly 0, and a block moved by half its norm reads exactly 0.5."""
    blocks = EncoderDrift(_checkpoint(tmp_path)).blocks(_stock())

    assert blocks[0] == pytest.approx(0.0)
    assert blocks[1] == pytest.approx(0.5)


def test_blocks_refuses_a_comparison_that_matches_nothing(tmp_path: Path) -> None:
    """The failure this module first reported as a drift of 0.000 must raise instead."""
    with pytest.raises(ValueError, match="nothing was compared"):
        EncoderDrift(_checkpoint(tmp_path)).blocks({"nothing.like.ours": torch.ones(4)})


def test_pairs(tmp_path: Path) -> None:
    """The shared arithmetic both drift columns are built from, as squared norms."""
    measured = EncoderDrift(_checkpoint(tmp_path)).pairs(_stock())
    pairs = {block: (changed, original) for block, changed, original in measured}

    assert pairs[1] == (pytest.approx(4.0), pytest.approx(16.0)), "block 1 moved 1.0 in each of four cells"


def test_changes(tmp_path: Path) -> None:
    """Absolute drift carries no denominator, which is the whole reason it is reported beside the ratio."""
    member = EncoderDrift(_checkpoint(tmp_path))

    assert member.changes(_stock())[1] == pytest.approx(2.0), "four cells moved by 1.0 each"
    assert member.changes(_stock())[0] == pytest.approx(0.0)
    assert member.blocks(_stock())[1] == pytest.approx(0.5), "the same change over a stock norm of 4.0"


def _prefit_stock() -> dict[str, torch.Tensor]:
    """Stock weights the member did NOT start from: block 0 differs too, as an SSL prefix makes it."""
    return {**_stock(), "encoder.layer.0.weight": torch.full((4,), 2.0)}


def test_opened(tmp_path: Path) -> None:
    """The window comes off the member, and a checkpoint that never recorded one is refused."""
    assert EncoderDrift(_checkpoint(tmp_path, unfreeze_last=3)).opened() == (3, 0)
    assert EncoderDrift(_adapted(tmp_path)).opened() is None, "no window recorded is a refusal, not a zero"


def test_concentration(tmp_path: Path) -> None:
    """All the drift sits in block 1, and the window decides whether that is inside or outside it."""
    drift = EncoderDrift(_checkpoint(tmp_path))
    blocks = drift.blocks(_stock())

    assert drift.concentration(blocks, (1, 0)) == pytest.approx(1.0), "the top block is the opened one"
    assert drift.concentration(blocks, (1, 1)) == pytest.approx(0.0), "offset 1 opens block 0, which never moved"


def test_concentration_of_no_blocks(tmp_path: Path) -> None:
    assert EncoderDrift(_checkpoint(tmp_path)).concentration({STEM: 0.3}, (1, 0)) == pytest.approx(0.0)


def test_constructed(tmp_path: Path) -> None:
    """A share of 1.0 is a measurement only when the frozen blocks could have moved and did not."""
    member = EncoderDrift(_checkpoint(tmp_path))

    assert member.constructed(member.blocks(_stock()), (1, 0)), "fitted from stock: block 0 is bit-identical"
    assert not member.constructed(member.blocks(_prefit_stock()), (1, 0)), "an SSL prefix moves the frozen blocks"


def test_report(tmp_path: Path) -> None:
    """Reports what it measured, and hands the profile back rather than only logging it."""
    reported = EncoderDrift(_checkpoint(tmp_path)).report(_stock())

    assert reported[1] == pytest.approx(0.5)
