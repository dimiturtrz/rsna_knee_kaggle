"""Substituting a training run's own holdout array for an inference pass — and refusing to when it differs.

The refusals carry the weight. Accepting a record computed under another reading would score a member as
a function it was never read as, silently, in the tool that decides what we submit.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from kneemri.inference.holdout_slab import HoldoutSlab
from kneemri.inference.members import Member
from kneemri.inference.test_time import SpatialViews
from kneemri.inference.trainer_record import TrainerRecord
from kneemri.runs.holdout_record import FOREIGN

SLICES = 6
PLAIN = SpatialViews(degrees=(0.0,), shifts=((0.0, 0.0),))


def test_instead_of(tmp_path: Path, recorded: Callable[..., Member], unreadable_slab: Callable[[int], HoldoutSlab]):
    """The recorded array comes back in the CALLER's study order, not the record's."""
    member = recorded(tmp_path, ["s1", "s2", "s3"])
    window = member.window(SLICES, overlap=False)

    values = TrainerRecord(PLAIN).instead_of(member, unreadable_slab(SLICES), window, ["s3", "s1"])

    assert values is not None
    assert values == pytest.approx(np.array([[0.4, 0.5], [0.0, 0.1]]))


def test_instead_of_refuses_a_reading_the_trainer_never_took(
    tmp_path: Path,
    reading: Callable[[Path], Member],
    recorded: Callable[..., Member],
    unreadable_slab: Callable[[int], HoldoutSlab],
):
    """Overlap, rotation, a missing study and a foreign decode are four different functions, not this one.

    Each is checked separately because each fails for its own reason, and a substitution that accepted
    any one of them would score a member as something it was never read as.
    """
    member = recorded(tmp_path, ["s1", "s2"])
    slab = unreadable_slab(SLICES)
    window = member.window(SLICES, overlap=False)
    foreign, unrecorded = tmp_path / "foreign", tmp_path / "unrecorded"
    foreign.mkdir()
    unrecorded.mkdir()

    assert TrainerRecord(PLAIN).instead_of(member, slab, member.window(SLICES, overlap=True), ["s1"]) is None
    assert TrainerRecord(SpatialViews()).instead_of(member, slab, window, ["s1"]) is None
    assert TrainerRecord(PLAIN).instead_of(member, slab, window, ["s1", "absent"]) is None
    assert TrainerRecord(PLAIN).instead_of(recorded(foreign, ["s1"], FOREIGN), slab, window, ["s1"]) is None
    assert TrainerRecord(PLAIN).instead_of(reading(unrecorded), slab, window, ["s1"]) is None
