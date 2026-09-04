"""The geometry half of mounting a SAM encoder — the half a unit test is allowed to reach.

`SamVisionWeights.build` fetches a 375 MB checkpoint, which the `small` gate forbids here and which the
integration suite covers instead. What is testable is the re-grid, and it is also the part worth testing:
it is the only step that can be wrong while every shape still agrees.
"""

import torch

from kneemri.models.sam_trunk import MEDSAM_VIT_B, SamVisionWeights, TokensInsteadOfNeck


def test_grid():
    """Tokens a side is the read cut into patches — the number `regridded` is asked for."""
    assert SamVisionWeights("repo", read_size=336).grid == 21
    assert SamVisionWeights("repo", read_size=1024).grid == 64
    assert MEDSAM_VIT_B.grid == 21, "the pinned read, resampled from SAM's pretrained 64x64 table"


def test_regridded():
    """The table is resampled over its two SPATIAL axes and not across its feature channels.

    The shape assertion alone would pass for a permutation-free implementation that interpolated the
    stored (n, h, w, c) layout directly, because 768 columns resampled to 768 columns is a no-op on that
    axis and the result would still come back (1, 21, 21, 768). So the test also holds a table that is
    CONSTANT over space and varies over channels: resampling space cannot change it, and interpolating
    the wrong axes would.
    """
    embed = torch.randn(1, 64, 64, 768, generator=torch.Generator().manual_seed(0))

    resized = SamVisionWeights.regridded(embed, 21)

    assert resized.shape == (1, 21, 21, 768)
    assert resized.is_contiguous(), "the encoder assigns this straight onto pos_embed.data"

    channels = torch.arange(768, dtype=torch.float32)
    flat = channels.expand(1, 64, 64, 768).contiguous()

    held = SamVisionWeights.regridded(flat, 21)

    torch.testing.assert_close(held, channels.expand(1, 21, 21, 768), rtol=1e-5, atol=1e-5)


def test_medsam_vit_b():
    """The declared entry names the mirror that exists, at the read the pin is stated for."""
    assert MEDSAM_VIT_B.repository == "flaviagiammarino/medsam-vit-base"
    assert MEDSAM_VIT_B.read_size == 336
    assert MEDSAM_VIT_B.read_size % MEDSAM_VIT_B.patch == 0, "a read the patch grid cannot tile has no grid"


def test_forward():
    """The read-out permutes to a feature MAP, which is the contract the head's adapter reads.

    Channels-last in, channels-first out. The assertion is on the VALUES and not only the shape,
    because a wrong permutation of a square grid — swapping h and w rather than moving the channel
    axis — returns the right shape from the wrong tensor.
    """
    hidden = torch.arange(2 * 3 * 3 * 4, dtype=torch.float32).reshape(2, 3, 3, 4)

    mapped = TokensInsteadOfNeck.forward(hidden)

    assert mapped.shape == (2, 4, 3, 3)
    assert torch.equal(mapped, hidden.permute(0, 3, 1, 2))
