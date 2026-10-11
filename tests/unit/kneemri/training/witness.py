"""The optimiser witnesses, exercised against groups built by hand rather than by a trainer.

`tests/unit/kneemri/training/trainer.py` already proves the end-to-end path — that an `Adaptation`
reaches `AdamW` and the witness reads it back. These tests are the other half: the witness's own
arithmetic over the equivalence classes of group layout, which a trainer cannot produce on demand
(a single group, a frozen backbone, a decayed pair).
"""

import pytest
import torch

from kneemri.models.encoder_surgery import NO_LAYER_DECAY
from kneemri.training.witness import OptimiserWitness

ANCHOR = 9.8e-5
HEAD_DECAY = 0.037
HEAD_RATE = 1e-3


def _witness(*backbone_rates: float, decay: float = HEAD_DECAY, anchor: float = ANCHOR) -> OptimiserWitness:
    """A REAL optimiser over throwaway parameters, because the witness's subject is what torch built.

    A stub carrying a `param_groups` list would pass every assertion here and still let a change in how
    torch names or nests those keys go unnoticed — which is the only way these three methods can break.
    """
    rates = [HEAD_RATE, *backbone_rates]
    groups = [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": rate} for rate in rates]
    return OptimiserWitness(torch.optim.SGD(groups, weight_decay=decay), anchor)


def test_weight_decay():
    """Read off the HEAD group, which exists in every configuration including LoRA's empty backbone."""
    assert _witness(ANCHOR, ANCHOR).weight_decay() == pytest.approx(HEAD_DECAY)
    assert _witness().weight_decay() == pytest.approx(HEAD_DECAY), "no backbone group is still a decay"


def test_encoder_rate_decay():
    """The ratio between the top two backbone groups, and `NO_LAYER_DECAY` where no pair exists.

    The boundary is the whole point: one group and zero groups must both answer rather than raise on a
    short slice, because those are the layouts every flat arm in the ledger builds.
    """
    assert _witness(ANCHOR * 0.5, ANCHOR).encoder_rate_decay() == pytest.approx(0.5)
    assert _witness(ANCHOR).encoder_rate_decay() == NO_LAYER_DECAY, "one group is no pair"
    assert _witness().encoder_rate_decay() == NO_LAYER_DECAY, "and neither is none"


def test_backbone_rate():
    """The captured anchor, returned whatever the groups now say — a schedule rewrites them every step."""
    rewritten = _witness(ANCHOR * 0.04, ANCHOR * 0.04)
    assert rewritten.backbone_rate() == pytest.approx(ANCHOR), "the flag, not what OneCycleLR left behind"
    assert rewritten.encoder_rate_decay() == pytest.approx(1.0), "which the ratio alone could never say"
