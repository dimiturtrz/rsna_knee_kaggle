"""The drift gate's arithmetic, on hand-built state dicts where the right answer is known by hand."""

from pathlib import Path

import pytest
import torch

from kneemri.evaluation.encoder_drift import STEM, EncoderDrift
from kneemri.models.backbone import SMALL


def _checkpoint(into: Path, *, compiled: bool = False, variant: str = SMALL) -> Path:
    """A checkpoint shaped like a member's: a `model` state dict and the config the trainer records."""
    prefix = "backbone._orig_mod." if compiled else "backbone."
    path = into / "member.pt"
    torch.save(
        {
            "config": {"variant": variant},
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


def test_variant(tmp_path: Path) -> None:
    assert EncoderDrift(_checkpoint(tmp_path)).variant() == SMALL


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


def test_concentration(tmp_path: Path) -> None:
    """All the drift sits in block 1, which is inside the deepest four, so the share is 1.0."""
    drift = EncoderDrift(_checkpoint(tmp_path))

    assert drift.concentration(drift.blocks(_stock())) == pytest.approx(1.0)


def test_concentration_of_no_blocks(tmp_path: Path) -> None:
    assert EncoderDrift(_checkpoint(tmp_path)).concentration({STEM: 0.3}) == pytest.approx(0.0)


def test_report(tmp_path: Path) -> None:
    """Reports what it measured, and hands the profile back rather than only logging it."""
    reported = EncoderDrift(_checkpoint(tmp_path)).report(_stock())

    assert reported[1] == pytest.approx(0.5)
