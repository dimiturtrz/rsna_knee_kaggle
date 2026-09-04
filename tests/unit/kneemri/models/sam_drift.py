"""The drift arithmetic, on modules small enough to state the expected number by hand.

Nothing here fetches a checkpoint: the two SAM releases are the CALLER's job, and what has to be right
before that download is worth doing is the ratio between regions — which is pure tensor arithmetic over
two `named_parameters` listings and needs no weights at all to exercise.
"""

import pytest
import torch
from torch import nn

from kneemri.models.sam_drift import SamDrift


class _Pair(nn.Module):
    """Two named parameters whose norms are chosen so every expected ratio is exact."""

    def __init__(self, one: list[float], other: list[float]) -> None:
        super().__init__()
        self.one = nn.Parameter(torch.tensor(one))
        self.other = nn.Parameter(torch.tensor(other))


def test_moved() -> None:
    """Each parameter's movement is its distance divided by where it started."""
    before = _Pair([1.0, 0.0], [0.0, 2.0])
    after = _Pair([1.0, 0.5], [0.0, 3.0])

    moved = SamDrift.moved(before, after)

    assert moved["one"] == 0.5
    assert moved["other"] == 0.5


def test_moved_skips_a_parameter_the_two_releases_do_not_share() -> None:
    """A shape the baseline cannot match is left out of the census rather than raising."""
    before = _Pair([1.0, 0.0], [0.0, 2.0])
    after = _Pair([1.0, 0.5], [0.0, 3.0, 0.0])

    moved = SamDrift.moved(before, after)

    assert set(moved) == {"one"}


def test_moved_skips_a_parameter_that_started_at_zero() -> None:
    """A zero baseline has no relative movement to report, and dividing by it would print `inf`."""
    before = _Pair([0.0, 0.0], [0.0, 2.0])
    after = _Pair([1.0, 0.0], [0.0, 3.0])

    moved = SamDrift.moved(before, after)

    assert set(moved) == {"other"}


def test_region() -> None:
    """A block keeps its index; everything else is named by its top component alone."""
    assert SamDrift.region("layers.3.attn.qkv.weight") == "layers.3"
    assert SamDrift.region("neck.conv1.weight") == "neck"
    assert SamDrift.region("pos_embed") == "pos_embed"


def test_grouped() -> None:
    """A region pools its parameters as a root-mean-square, not as a mean."""
    grouped = SamDrift.grouped({"neck.conv1.weight": 0.0, "neck.conv2.weight": 1.0})

    # The mean of the two is 0.5; the root-mean-square is 0.707, and the gap between those two numbers
    # is the whole reason for pooling this way rather than averaging.
    assert grouped["neck"] == pytest.approx(0.70710678)


def test_lines() -> None:
    """The report counts what it compared and puts the region that moved most first."""
    lines = SamDrift.lines({"neck.conv1.weight": 0.9, "layers.0.attn.qkv.weight": 0.1})

    assert lines[0] == "2 parameters compared across 2 regions"
    assert lines[1] == "neck: 0.9000"
    assert lines[2] == "layers.0: 0.1000"
