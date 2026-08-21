import torch

from kneemri.models.depth_window import DepthWindow

SLOTS = 2
DEPTH = 4


def test_channels():
    """(i-1, i, i+1) along the slot's own depth, clamped at its ends."""
    imgs = torch.arange(2 * SLOTS * DEPTH, dtype=torch.uint8).reshape(2, SLOTS, DEPTH, 1, 1)

    window = DepthWindow.channels(imgs)

    assert window.shape == (2, SLOTS, DEPTH, len(DepthWindow.NEIGHBOURS), 1, 1)
    assert torch.equal(window[:, :, :, 1], imgs), "the centre channel must be the slice being encoded"
    assert torch.equal(window[:, :, 1, 0], imgs[:, :, 0]), "channel 0 is the slice before"
    assert torch.equal(window[:, :, 1, 2], imgs[:, :, 2]), "channel 2 is the slice after"
    assert torch.equal(window[:, :, 0, 0], imgs[:, :, 0]), "the first slice clamps rather than wraps"
    assert torch.equal(window[:, :, -1, 2], imgs[:, :, -1]), "and so does the last"


def test_channels_never_reach_across_a_slot():
    """The slot axis holds different acquisitions, so a slot's window must be a function of it alone."""
    imgs = torch.arange(SLOTS * DEPTH, dtype=torch.uint8).reshape(1, SLOTS, DEPTH, 1, 1)

    window = DepthWindow.channels(imgs)

    for slot in range(SLOTS):
        alone = DepthWindow.channels(imgs[:, slot : slot + 1])
        assert torch.equal(window[:, slot], alone[:, 0]), "a slot's window moved when its neighbours were present"
