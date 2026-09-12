"""`Adaptation` carries the defaults every ledger number was measured under — pinned, not described."""

import pytest

from kneemri.models.encoder_surgery import NO_OFFSET
from kneemri.models.model import NO_LAYER_DECAY
from kneemri.training.adaptation import WEIGHT_DECAY, Adaptation

RATIO = 125
"""Head rate over backbone rate at the inherited default — the one number the docstring argues for."""


def test_adaptation_defaults_are_the_recipe_every_arm_ran() -> None:
    """PINNED, because a changed default silently re-prices every number already in the ledger."""
    adaptation = Adaptation()
    assert adaptation.head_lr / adaptation.backbone_lr == pytest.approx(RATIO)
    assert adaptation.weight_decay == pytest.approx(WEIGHT_DECAY)
    assert adaptation.unfreeze_last == Adaptation().unfreeze_last
    assert adaptation.unfreeze_offset == NO_OFFSET
    assert adaptation.layer_decay == pytest.approx(NO_LAYER_DECAY)


def test_adaptation_is_frozen() -> None:
    """A run's recipe may not be edited after the optimiser groups have been built from it."""
    with pytest.raises(AttributeError):
        Adaptation().backbone_lr = 1.0  # type: ignore[misc]
