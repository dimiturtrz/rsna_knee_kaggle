"""The compositions the loss flags refuse, checked without a card, a corpus or an encoder."""

import pytest

from kneemri.training.config_tree import TrainConfig
from kneemri.training.objective_checks import ObjectiveChecks

BATCH = 12


def _checks(**asked: object) -> ObjectiveChecks:
    """One run's flags and the guard over them, which is the pair every test here is about."""
    return ObjectiveChecks(TrainConfig().with_leaves(**{"batch": BATCH, **asked}))  # type: ignore[arg-type]


def test_composed():
    """The house recipe passes, and each refusal names the flag whose term would have been lost."""
    _checks(rank_weight=1.0).composed()

    for asked, named in (
        ({"auc_margin": 0.8, "rank_weight": 1.0}, "--auc-margin"),
        ({"dissent_weight": 0.3, "distill_weight": 0.3}, "--dissent-weight"),
        ({"contrast_weight": 1.0, "distill_weight": 0.3}, "--contrast-weight"),
        ({"feature_weight": 0.3, "dissent_weight": 0.3}, "--feature-weight"),
        ({"pair_memory": 64}, "--pair-memory"),
        ({"rank_weight": 1.0, "batch": 1}, "--batch 1"),
    ):
        with pytest.raises(ValueError, match=named) as refusal:
            _checks(**asked).composed()
        assert (
            "not composed" in str(refusal.value)
            or "nothing for it" in str(refusal.value)
            or "zero" in str(refusal.value)
        ), "a refusal says what would have been silently dropped, not only that it refused"
