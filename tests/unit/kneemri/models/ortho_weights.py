"""The mirror for the checkpoint loader — the one half that can be checked without their weights."""

import torch

from kneemri.models.ortho_weights import DENOISER_PREFIX, DenoiserWeights


def test_stripped() -> None:
    """The wrapper's prefix comes off its parameters and its own schedule buffers are dropped."""
    state = {
        f"{DENOISER_PREFIX}out.weight": torch.zeros(1),
        "sqrt_alphas_cumprod": torch.zeros(1),
    }
    stripped = DenoiserWeights.stripped(state)
    assert list(stripped) == ["out.weight"]


def test_stripped_is_empty_for_an_unprefixed_checkpoint() -> None:
    """An empty result means "not prefixed", which is why the caller falls back to the whole state."""
    assert DenoiserWeights.stripped({"out.weight": torch.zeros(1)}) == {}
