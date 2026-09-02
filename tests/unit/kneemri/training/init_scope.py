"""How much of a checkpoint a warm start takes."""

import pytest
import torch

from kneemri.training.init_scope import InitScope, TrunkOnly, WholeModel

TRUNK = "encoder.blocks.0.weight"
SLOTS = "head.slot_emb"


def _state() -> dict[str, torch.Tensor]:
    """A checkpoint's parameters in miniature: one trunk tensor, one head tensor shaped by slot count."""
    return {TRUNK: torch.zeros(2, 2), SLOTS: torch.zeros(3, 4)}


def test_named():
    """A scope is chosen by name, and a name that does not exist raises before the card is spent."""
    assert isinstance(InitScope.named("all"), WholeModel)
    assert isinstance(InitScope.named("trunk"), TrunkOnly)
    with pytest.raises(KeyError):
        InitScope.named("head")


def test_whole_model_carried():
    """LP-FT's case takes everything: stage one's head IS the head stage two wants."""
    assert set(WholeModel().carried(_state())) == {TRUNK, SLOTS}


def test_whole_model_dropped():
    """Nothing is deliberately left cold, so any absent key is still the failure it always was."""
    assert WholeModel().dropped(_state()) == set()


def test_trunk_only_carried():
    """The cross-corpus case takes the representation and leaves the foreign head behind.

    MRNet's three series make `head.slot_emb` (3, 256) against this corpus's (6, 256), so a whole-model
    load raises — after stage one has already spent its card.
    """
    assert set(TrunkOnly().carried(_state())) == {TRUNK}


def test_trunk_only_dropped():
    """What was left cold is NAMED, so a load can tell it from a checkpoint that is simply short."""
    assert TrunkOnly().dropped(_state()) == {SLOTS}
