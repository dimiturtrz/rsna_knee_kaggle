from typing import cast

import pytest
import torch

from core.findings import TARGETS
from kneemri.models.slice_pooling import ATTEND, CHANNELS, DEPTH, LABELWISE, LABELWISE_PATCH, MAXIMUM, SLICE_POOLINGS

DIM = 8
BATCH, SLOTS, GROUP = 2, 3, 4


@pytest.mark.parametrize(
    ("name", "per_slice"),
    [(CHANNELS, False), (MAXIMUM, True), (ATTEND, True), (DEPTH, True), (LABELWISE, True)],
)
def test_slice_poolings_partition_the_vocabulary_by_what_the_encoder_sees(*, name: str, per_slice: bool):
    """ONE REP PER CLASS of the first decision a name makes: is a slice encoded alone, or stacked?

    `channels` is the whole of the FALSE class — its three slices become the encoder's three channels, so
    a slot costs one forward and its physical thickness is coupled to the slice count. Every other name
    is in the TRUE class: a forward per slice, thickness and count decoupled, and a slice axis left over
    for an aggregator to remove. The classes are asserted by NAME because this is the fact three call
    sites each read out of this table — `options.py` builds its `--slice-pooling` choices from these keys
    and `members.py` reads the string back out of a manifest.
    """
    assert SLICE_POOLINGS[name].per_slice is per_slice


def test_slice_poolings_reject_a_name_outside_the_vocabulary():
    """THE BOUNDARY, and the failure it exists for: an unknown name must not resolve to anything.

    It used to. `per_slice` asked `in {"max", "attend"}`, so a stale or misspelled name fell through to
    channel stacking and rebuilt a different function on different pixels without complaint — the
    plumbing failure this project keeps paying for. A lookup raises where the model is built instead.
    """
    with pytest.raises(KeyError):
        _ = SLICE_POOLINGS["mean"]


@pytest.mark.parametrize("name", [CHANNELS, MAXIMUM])
def test_slice_poolings_carry_no_aggregator_when_nothing_is_learned(name: str):
    """The class with NO trainable pooling: a stacked slot has no slice axis, and `max` needs no module.

    Kept as its own assertion rather than folded into the behavioural test below because `None` here is
    what tells `StudyModel` to fall back to `amax` — the absence is the instruction, not an omission.
    """
    assert SLICE_POOLINGS[name].aggregator is None


@pytest.mark.parametrize(("name", "sees_order"), [(ATTEND, False), (DEPTH, True)])
def test_slice_poolings_build_an_aggregator_that_pools_the_slice_axis_away(*, name: str, sees_order: bool):
    """The class WITH a trainable pooling, split by the axis the two members differ on.

    Tested by building the module rather than by comparing class objects, because what the table promises
    its consumers is a callable that turns "b s g d" into "b s d" — a record naming a class with a
    different signature would satisfy an identity check and then fail at model-build time.

    ORDER is the partition inside this class and the reason both members exist: `attend` scores each
    slice from that slice alone and so cannot represent that a finding continues into the next slice;
    `depth` mixes a slice with its neighbours first and can. Asserted against a hand-set neighbour tap,
    since at initialisation `depth` is deliberately the identity and the two are then indistinguishable.
    """
    aggregator = SLICE_POOLINGS[name].aggregator
    assert aggregator is not None
    torch.manual_seed(2026)
    pool = aggregator(DIM).eval()
    if sees_order:
        with torch.no_grad():
            cast("torch.nn.Conv1d", pool.mix).weight[:, :, 0] = 0.5

    x = torch.randn(BATCH, SLOTS, GROUP, DIM)
    with torch.no_grad():
        out = pool.forward(x)
        reversed_out = pool.forward(x.flip(dims=(2,)))

    assert out.shape == (BATCH, SLOTS, DIM), "an aggregator's job is to remove the slice axis"
    order_blind = bool(torch.allclose(out, reversed_out, atol=1e-4))
    assert order_blind == (not sees_order), f"{name} reads slice order as {not order_blind}, expected {sees_order}"


def test_slice_poolings_may_replace_the_slice_axis_with_a_diagnosis_axis():
    """The third class of aggregator, and the one whose OUTPUT SHAPE differs from the other two.

    `attend` and `depth` remove the slice axis; `labelwise` replaces it with a per-finding axis, so the
    head receives twelve readings of a slot instead of one. Asserted here because this table is what
    tells `StudyModel` and `TrialHead` what to expect, and a pooling whose rank surprises them is the
    plumbing failure this module exists to prevent — one that a shape error would only surface deep
    inside a head, or not at all.
    """
    aggregator = SLICE_POOLINGS[LABELWISE].aggregator
    assert aggregator is not None
    torch.manual_seed(2026)
    with torch.no_grad():
        out = aggregator(DIM).eval()(torch.randn(BATCH, SLOTS, GROUP, DIM))
    assert out.shape == (BATCH, SLOTS, len(TARGETS), DIM)


def test_build():
    """`build` hands the trunk's token width only to a pooling that reads patches, and None to `max`."""
    torch.manual_seed(2026)
    assert SLICE_POOLINGS[MAXIMUM].build(DIM, DIM // 2) is None
    with torch.no_grad():
        x = torch.randn(BATCH, SLOTS, GROUP, DIM)
        labelwise = SLICE_POOLINGS[LABELWISE].build(DIM, DIM // 2)
        patch = SLICE_POOLINGS[LABELWISE_PATCH].build(DIM, DIM // 2)
        assert labelwise is not None
        assert patch is not None
        patches = torch.randn(BATCH, SLOTS, GROUP, 6, DIM // 2)
        assert patch(x, patches).shape == labelwise(x).shape == (BATCH, SLOTS, len(TARGETS), DIM)
    assert SLICE_POOLINGS[LABELWISE_PATCH].reads_patches
    assert not SLICE_POOLINGS[LABELWISE].reads_patches
