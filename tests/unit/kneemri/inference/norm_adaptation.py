"""What the LayerNorm-affine adapt pass must do, and the two things it must never do."""

from functools import partial
from typing import override

import pytest
import torch
from torch import Tensor, nn

from kneemri.inference.norm_adaptation import (
    CARDINALITY_BOUNDS,
    NEGATIVE_WEIGHT,
    CardinalityAwareLN,
    NoAdaptation,
    NotAdaptableError,
)

LABELS = 12
WINDOWS = 4
WIDTH = 8
NORM_BLOCKS = 6
QUARTER = 0.25
LOUD = 2.0
LOUDER = 4.0
QUIET = -8.0


class _Member(nn.Module):
    """A member shaped like ours: several LayerNorm blocks and a readout onto our label count."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.LayerNorm(WIDTH)
        self.rest = nn.ModuleList([nn.LayerNorm(WIDTH) for _ in range(NORM_BLOCKS - 1)])
        self.head = nn.Linear(WIDTH, LABELS)

    @override
    def forward(self, batch: Tensor) -> Tensor:
        """One row per study, as a real member emits: the window axis is pooled INSIDE the member."""
        hidden = self.first(batch)
        for block in self.rest:
            hidden = block(hidden)
        return self.head(hidden.mean(dim=-2))


@pytest.fixture(name="member")
def _member() -> _Member:
    torch.manual_seed(2026)
    return _Member()


@pytest.fixture(name="batch")
def _batch() -> Tensor:
    torch.manual_seed(2026)
    return torch.randn(1, WINDOWS, WIDTH)


def test_cardinality_aware_l_n_on(member: _Member, batch: Tensor) -> None:
    """The episodic reset is the load-bearing half: adapting study N must not reach study N+1."""
    before = [parameter.detach().clone() for parameter in member.parameters()]
    with CardinalityAwareLN().on(member, partial(member, batch)):
        pass
    for parameter, original in zip(member.parameters(), before, strict=True):
        assert torch.equal(parameter, original)


def test_on_moves_the_affines_while_it_is_open(member: _Member, batch: Tensor) -> None:
    """Inside the block the member must actually be different, or the pass is a no-op wearing a name."""
    original = member.first.weight.detach().clone()
    with CardinalityAwareLN(lr=1e-1).on(member, partial(member, batch)):
        assert not torch.equal(member.first.weight, original)


def test_on_restores_even_when_the_read_raises(member: _Member, batch: Tensor) -> None:
    """A failed read must not leave an adapted member behind for every study after it."""
    original = member.first.weight.detach().clone()
    with pytest.raises(ZeroDivisionError), CardinalityAwareLN(lr=1e-1).on(member, partial(member, batch)):
        raise ZeroDivisionError
    assert torch.equal(member.first.weight, original)


def test_no_adaptation_on(member: _Member, batch: Tensor) -> None:
    """The control every reading is against serves the trained function untouched."""
    before = [parameter.detach().clone() for parameter in member.parameters()]
    with NoAdaptation().on(member, partial(member, batch)):
        for parameter, original in zip(member.parameters(), before, strict=True):
            assert torch.equal(parameter, original)


def test_affines(member: _Member) -> None:
    """SAR freezes the last three blocks; the head is never adapted by any published recipe."""
    adapted = CardinalityAwareLN().affines(member)
    assert len(adapted) == 2 * (NORM_BLOCKS - 3)
    assert all(any(parameter is one for one in adapted) is False for parameter in member.head.parameters())


def test_affines_refuses_a_member_it_cannot_reach() -> None:
    """Serving unadapted under an adapted name is the failure this raise exists to prevent."""
    with pytest.raises(NotAdaptableError):
        CardinalityAwareLN().affines(object())


def test_cardinality() -> None:
    """Our p5 is zero — a clean knee must be allowed to stay clean — and our p95 is eight."""
    low, high = CARDINALITY_BOUNDS
    recipe = CardinalityAwareLN()
    studies = torch.stack([torch.zeros(LABELS), torch.ones(LABELS), torch.full((LABELS,), QUARTER)])
    counts = recipe.cardinality(studies)
    assert [int(count) for count in counts] == [low, high, round(QUARTER * LABELS)]


def test_wanted() -> None:
    """Each study gets its OWN top-k̂; a batch must never share one, or a neighbour leaks into a reading."""
    quiet = torch.full((LABELS,), 1e-3)
    loud = quiet.clone()
    loud[:4] = 0.9
    chosen = CardinalityAwareLN().wanted(torch.stack([quiet, loud]))
    assert int(chosen[0].sum()) == 0
    assert bool(chosen[1, :4].all()) and int(chosen[1].sum()) == 4


def test_loss() -> None:
    """The objective must reward a sharper version of the study's own reading, not a different one."""
    flat = torch.zeros(1, LABELS)
    flat[:, :3] = LOUD
    sharper = flat.clone()
    sharper[:, :3] = LOUDER
    sharper[:, 3:] = -LOUD
    recipe = CardinalityAwareLN()
    assert float(recipe.loss(sharper)) < float(recipe.loss(flat))


def test_loss_means_each_term_over_its_own_set() -> None:
    """CARVE Eq. 9 divides by |P| and |N|, not by the label count — the difference is the whole weight.

    A study whose top-k̂ is small gets a sharpening term meaned over those few labels. Dividing by the
    twelve instead would shrink it by |P|/L and leave the objective mostly suppression, which is the
    draft this replaced. Priced here against the arithmetic rather than against a remembered shape.
    """
    reading = torch.zeros(1, LABELS)
    reading[:, :3] = LOUD
    recipe = CardinalityAwareLN()
    probability = torch.sigmoid(reading.float())
    wanted = recipe.wanted(probability)

    sharpen = float(-(torch.log(probability) * wanted).sum() / int(wanted.sum()))
    suppress = float(-(torch.log1p(-probability) * ~wanted).sum() / int((~wanted).sum()))

    assert float(recipe.loss(reading)) == pytest.approx(sharpen + NEGATIVE_WEIGHT * suppress, abs=1e-5)


def test_loss_suppresses_everything_when_the_study_reads_clean() -> None:
    """At cardinality zero there is no label to sharpen, so the whole objective is suppression."""
    quiet = torch.full((1, LABELS), QUIET)
    recipe = CardinalityAwareLN()
    assert int(recipe.cardinality(torch.sigmoid(quiet))) == 0
    assert float(recipe.loss(quiet)) == pytest.approx(0.0, abs=1e-3)
