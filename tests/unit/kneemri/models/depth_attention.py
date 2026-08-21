from typing import cast

import torch

from kneemri.models.depth_attention import DepthAttention

DIM = 8
BATCH, SLOTS, GROUP = 2, 3, 4


def test_forward():
    """At initialisation the mixer is the identity tap, so an arm's first step is the incumbent's.

    This is what makes the comparison clean: a randomly initialised mixer would begin by smearing the
    features it was handed, and the arm would be paying that cost against the mechanism's account.
    """
    torch.manual_seed(2026)
    pool = DepthAttention(DIM).eval()
    x = torch.randn(BATCH, SLOTS, GROUP, DIM)

    with torch.no_grad():
        out = pool.forward(x)

    assert out.shape == (BATCH, SLOTS, DIM), "the slice axis is pooled away"
    assert torch.allclose(out, x.mean(dim=2), atol=1e-5), (
        "identity mix plus a zeroed gate is exactly `attend` at step zero"
    )


def test_forward_cannot_mix_across_the_slot_boundary():
    """THE BOUNDARY CASE of the depth axis: a slot's edge slice has a neighbour only INSIDE its slot.

    The slots are folded into the batch precisely so the kernel cannot run across the boundary — a slot
    is a different acquisition of the same knee, and a tap reaching over would mix a sagittal slice into
    an axial one purely because of how the tensor is laid out. Asserted by changing ONE slot's pixels and
    requiring every other slot's output to be bit-identical; a kernel that crossed would leak into the
    neighbouring slot's edge slice and this would fail.
    """
    torch.manual_seed(2026)
    pool = DepthAttention(DIM).eval()
    with torch.no_grad():
        cast("torch.nn.Conv1d", pool.mix).weight[:, :, 0] = 0.5
    x = torch.randn(BATCH, SLOTS, GROUP, DIM)
    perturbed = x.clone()
    perturbed[:, 0] += 10.0

    with torch.no_grad():
        out, moved = pool.forward(x), pool.forward(perturbed)

    assert torch.equal(out[:, 1:], moved[:, 1:]), "one slot's slices may not reach into another slot's"
    assert not torch.allclose(out[:, 0], moved[:, 0]), "and the slot that did change must change"


def test_forward_sees_slice_order():
    """THE WHOLE POINT: once a neighbour tap is non-zero, reversing the slices changes the answer.

    Written against a hand-set kernel rather than a trained one because the claim is about what the
    operator CAN express, and training it here would test an optimiser. The paired assertion on
    `SliceAttention` is `test_forward_cannot_see_slice_order`; the two together are the finding.
    """
    torch.manual_seed(2026)
    pool = DepthAttention(DIM).eval()
    with torch.no_grad():
        cast("torch.nn.Conv1d", pool.mix).weight[:, :, 0] = 0.5
    x = torch.randn(BATCH, SLOTS, GROUP, DIM)

    with torch.no_grad():
        assert not torch.allclose(pool.forward(x), pool.forward(x.flip(dims=(2,))), atol=1e-4), (
            "a mixer that reaches its neighbour must distinguish a stack from its reverse"
        )
