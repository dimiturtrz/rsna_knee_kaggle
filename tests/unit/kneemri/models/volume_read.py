from types import SimpleNamespace

import pytest
import torch

from kneemri.models.token_pooling import CLS_MEAN, MEAN
from kneemri.models.volume_read import VolumeRead

TUBELET = 2
PATCH = 4
NATIVE = 8
FRAMES = 4
VIDEOS = 3
WIDTH = 8

VOLUME = VolumeRead(TUBELET, PATCH, NATIVE)
TOKENS = (NATIVE // PATCH) ** 2 * (FRAMES // TUBELET)


def _config(tubelet: int = TUBELET) -> SimpleNamespace:
    return SimpleNamespace(tubelet_size=tubelet, patch_size=PATCH, crop_size=NATIVE)


def test_of():
    """A tubelet in the config is what makes a trunk volumetric, and the absence of one is the 2D path."""
    assert VolumeRead.of(_config(), MEAN, "channels") == VOLUME
    assert VolumeRead.of(_config(tubelet=0), CLS_MEAN, "channels") is None
    assert VolumeRead.of(None, CLS_MEAN, "channels") is None, "a configless stand-in reads as 2D"

    with pytest.raises(ValueError, match="no class token"):
        VolumeRead.of(_config(), CLS_MEAN, "channels")

    with pytest.raises(ValueError, match="volumetric read"):
        VolumeRead.of(_config(), MEAN, "max")


def test_frames():
    """The flat stack the resize and the normalisation ran on, folded back into one video per slot."""
    flat = torch.zeros(VIDEOS * FRAMES, 3, NATIVE, NATIVE)
    folded = VOLUME.frames(flat, VIDEOS)
    assert folded.shape == (VIDEOS, FRAMES, 3, NATIVE, NATIVE)
    assert torch.equal(folded.reshape(flat.shape), flat), "folding must not reorder the pixels"


def test_check_tokens():
    """THE NEGATIVE CASE IS THE LOAD-BEARING HALF, and it is what killed the first version of this guard.

    That version predicted the token count from `x`'s own shape, which is the shape the trunk itself
    reads, so the two agreed at every size and it could not fail. Every raise below is a read the trunk
    would have accepted in silence.
    """
    native = torch.zeros(1, FRAMES, 3, NATIVE, NATIVE)
    tokens = torch.zeros(1, TOKENS, WIDTH)

    VOLUME.check_tokens(native, tokens)

    with pytest.raises(ValueError, match="fitted at"):
        VOLUME.check_tokens(torch.zeros(1, FRAMES, 3, NATIVE * 2, NATIVE * 2), tokens)

    with pytest.raises(ValueError, match="fitted at"):
        VOLUME.check_tokens(torch.zeros(1, FRAMES + 1, 3, NATIVE, NATIVE), tokens)

    with pytest.raises(ValueError, match="fitted at"):
        VOLUME.check_tokens(native, torch.zeros(1, TOKENS + 1, WIDTH))
