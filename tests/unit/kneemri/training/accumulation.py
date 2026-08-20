"""The four functions of two integers that turn micro-batches into optimiser steps."""

import pytest
import torch

from kneemri.training.accumulation import Accumulation


def test_share():
    """A micro-batch contributes its SHARE, or a group of two arrives at twice the magnitude.

    Gradients sum in the leaves between optimiser steps, so accumulation without the division is a
    doubled learning rate wearing a batch's name — the schedule every arm was tuned under, silently
    scaled by a flag that claims to change only memory.
    """
    loss = torch.tensor(3.0)
    assert float(Accumulation(1).share(loss)) == 3.0, "no accumulation must leave the loss untouched"
    assert float(Accumulation(3).share(loss)) == pytest.approx(1.0)


def test_closes():
    """The optimiser steps on the micro-batch that completes a group, and on no other."""
    every_second = Accumulation(2)
    assert [every_second.closes(step) for step in (1, 2, 3, 4)] == [False, True, False, True]
    assert all(Accumulation(1).closes(step) for step in (1, 2, 3)), "un-accumulated steps every time"


def test_groups():
    """Optimiser steps, not micro-batches — the number OneCycleLR must be sized against.

    A schedule counted in micro-batches is `every` times too long, so the run ends mid-decay having
    never reached the low rate the cycle exists to finish at.
    """
    assert Accumulation(2).groups(10) == 5
    assert Accumulation(3).groups(10) == 3, "the trailing partial group is dropped, not rounded up"
    assert Accumulation(4).groups(2) == 1, "fewer micro-batches than a group still makes one step"


def test_whole():
    """Micro-batches truncated to complete groups, so no backward pass is paid for and discarded."""
    assert Accumulation(2).whole(10) == 10
    assert Accumulation(3).whole(10) == 9
    assert Accumulation(1).whole(7) == 7, "without accumulation this is the identity"
