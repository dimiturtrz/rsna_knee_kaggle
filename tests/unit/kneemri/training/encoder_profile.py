from argparse import Namespace
from pathlib import Path

import pytest
import torch

from kneemri.training.encoder_profile import SavedAdaptation

MASK_RATIO = 0.15
RETENTION = 0.6
CACHE_KEY = "336px_12sl"


def _saved(cache: str = CACHE_KEY) -> SavedAdaptation:
    return SavedAdaptation(
        {
            "variant": "small",
            "cache": cache,
            "config": {"mask_ratio": MASK_RATIO, "patch": 14, "retention": RETENTION},
            "encoder": {},
        }
    )


def test_load(tmp_path: Path):
    """The file comes back with its recipe, which is the half a caller would otherwise have to retype."""
    checkpoint = tmp_path / "adapt.pt"
    torch.save(_saved().state, checkpoint)

    assert SavedAdaptation.load(checkpoint).state["cache"] == CACHE_KEY


def test_recipe():
    """Recorded values, shaped as the namespace an objective builder reads its flags from."""
    recipe = _saved().recipe

    assert isinstance(recipe, Namespace)
    assert recipe.mask_ratio == MASK_RATIO
    assert recipe.retention == RETENTION


def test_encoder():
    """The trunk is named by the checkpoint, so a profile cannot be read through the wrong one."""
    assert _saved().encoder is not None


def test_require_cache():
    """The defect this refuses is the one that already happened: weights fitted on other pixels.

    Two adaptations on disk were adapted on the default 12-slice cache while every arm importing them
    trained on 20. Matching keys pass silently; a mismatch is an error rather than a warning, because a
    profile read on a different slice population is a reading of a different experiment.
    """
    _saved().require_cache(CACHE_KEY)

    with pytest.raises(RuntimeError, match="different pixels"):
        _saved().require_cache("336px_20sl")
