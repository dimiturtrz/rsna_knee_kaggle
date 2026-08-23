import logging

import pytest
import torch
from torch import Tensor

from kneemri.kaggle.slot_fill import FILL_WITNESS_FLOOR, ZERO_SLOT_SHARE_MAX, SlotFill

_SLOTS = ("SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS")


def _presence(rows: list[list[float]]) -> Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def test_observed():
    """Filled slots, served studies and wholly-unrouted studies counted across several chunks."""
    witness = SlotFill(_SLOTS)

    witness.observed(_presence([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
    witness.observed(_presence([[1.0, 0.0, 0.0]]))

    assert witness.studies == 3, "every row of every chunk is a study"
    assert witness.empty == 1, "only the all-zero row routed to nothing"
    assert list(witness.filled) == [2, 1, 0], "per-slot counts accumulate across chunks"


def test_shares():
    """Fill share is per slot over every study served, not per chunk."""
    witness = SlotFill(_SLOTS)
    witness.observed(_presence([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))

    assert witness.shares() == {"SAG_FLUID_FS": 1.0, "COR_FLUID_FS": 0.5, "AX_FLUID_FS": 0.0}


def test_report(caplog: pytest.LogCaptureFixture):
    """Both halves reach the log: the per-slot shares and the count that routed to nothing."""
    witness = SlotFill(_SLOTS)
    witness.observed(_presence([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    with caplog.at_level(logging.INFO):
        witness.report()

    assert "slot fill SAG_FLUID_FS: 0.500" in caplog.text
    assert "studies with no slot at all: 1/2" in caplog.text


def test_report_says_so_when_the_decode_path_never_ran():
    """No studies is not a fill of zero — it is a witness that saw nothing, and it must not divide."""
    witness = SlotFill(_SLOTS)

    witness.report()

    assert witness.studies == 0, "and reporting it does not invent a denominator"


def test_refuse_if_collapsed():
    """A split routed mostly to nothing raises, so the run writes no table instead of a chance one."""
    witness = SlotFill(_SLOTS)
    witness.observed(_presence([[0.0, 0.0, 0.0]] * FILL_WITNESS_FLOOR))

    with pytest.raises(ValueError, match="routing contract broke"):
        witness.refuse_if_collapsed()


def test_refuse_if_collapsed_permits_a_split_inside_the_ceiling():
    """Below the legislated share it is data being hard, not the router being broken."""
    empty = int(FILL_WITNESS_FLOOR * ZERO_SLOT_SHARE_MAX)
    witness = SlotFill(_SLOTS)
    witness.observed(_presence([[0.0, 0.0, 0.0]] * empty + [[1.0, 0.0, 0.0]] * (FILL_WITNESS_FLOOR - empty)))

    witness.refuse_if_collapsed()

    assert witness.empty == empty, "the studies were still counted, they just did not trip the gate"


def test_refuse_if_collapsed_never_fires_on_the_visible_test_set():
    """THE THREE-STUDY PASS MUST SUCCEED. A gate that fails it costs every slot instead of saving one."""
    witness = SlotFill(_SLOTS)
    witness.observed(_presence([[0.0, 0.0, 0.0]] * 3))

    witness.refuse_if_collapsed()

    assert witness.studies < FILL_WITNESS_FLOOR, "observed and reported, but below the refusing floor"
