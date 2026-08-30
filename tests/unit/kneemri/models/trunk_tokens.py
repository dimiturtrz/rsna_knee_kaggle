"""The adapter that makes a non-ViT trunk answer the head's questions."""

from collections.abc import Callable

import pytest
import torch
from torch import nn

from kneemri.models.trunk_tokens import TokenOutput, TrunkAsTokens


def test_forward(feature_map: Callable[[int, int], nn.Module], token_sequence: Callable[[int, int], nn.Module]):
    """A feature map arrives as tokens with a class token in front, in row-major order.

    The head slices `tokens[:, 1:]` to get the patch grid and pools it. If the flatten dropped the
    prefix slot, that slice would silently eat the first spatial token and the model would train on a
    grid missing its top-left corner — a defect with no shape error to announce it.
    """
    emitted = TrunkAsTokens.from_map(feature_map(4, 3), 4).forward(torch.zeros(2, 3, 96, 96))
    tokens = emitted.last_hidden_state

    assert tokens.shape == (2, 1 + 9, 4), "one prefix token, then the 3x3 grid, at the trunk's own width"
    assert torch.equal(tokens[0, 1:, 0], torch.arange(9, dtype=torch.float32)), "row-major, like a ViT's patches"
    assert tokens[0, 0, 0] == pytest.approx(4.0), "the prefix holds the map's mean, which is what a CNN pools"

    same = TrunkAsTokens.from_sequence(token_sequence(4, 9), 4).forward(torch.zeros(2, 3, 96, 96))

    assert torch.equal(same.last_hidden_state, tokens), (
        "both constructions end at the same contract, whatever layout they started from"
    )


def test_from_map(feature_map: Callable[[int, int], nn.Module]):
    """The map arm is the one that moves axes, and moving them wrongly has no shape error to announce it."""
    adapted = TrunkAsTokens.from_map(feature_map(4, 3), 4)

    assert adapted.config.hidden_size == 4, "the width the trunk was declared at, not one read off a tensor"
    assert adapted.forward(torch.zeros(1, 3, 96, 96)).last_hidden_state.shape == (1, 1 + 9, 4), "(n, d, h, w) flattened"


def test_from_sequence(token_sequence: Callable[[int, int], nn.Module]):
    """The sequence arm moves nothing: the layout is already right and only the prefix slot is missing."""
    adapted = TrunkAsTokens.from_sequence(token_sequence(4, 9), 4)

    assert adapted.config.hidden_size == 4, "the width the trunk was declared at, not one read off a tensor"
    assert adapted.forward(torch.zeros(1, 3, 96, 96)).last_hidden_state.shape == (1, 1 + 9, 4), "one slot, no reshape"


class _Checkpointing(nn.Module):
    """A trunk that checkpoints, which is what the wrapper has to reach and read back.

    Local rather than a shared fixture: the two contract fakes exist to emit the two LAYOUTS, and this
    one emits nothing at all. It records instead — which is the only way an assert can tell a hook that
    travelled from one that stopped at the wrapper.
    """

    def __init__(self) -> None:
        super().__init__()
        self.asked: list[dict[str, object]] = []

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, object]) -> None:
        """What `transformers` exposes, recorded rather than acted on."""
        self.asked.append(gradient_checkpointing_kwargs)

    @property
    def is_gradient_checkpointing(self) -> bool:
        """The readback an HF model offers, answered from what was actually asked of this trunk."""
        return bool(self.asked)


def test_gradient_checkpointing_enable():
    """The hook has to reach the TRUNK, not stop at the wrapper wearing its contract.

    `EncoderSurgery.recompute_activations` calls this on whatever it holds as a backbone. An
    `nn.Module` raises AttributeError for a name that is not a submodule, so a wrapped trunk made
    `--recompute-activations` die at model construction — before a batch, and for every arm on a
    non-ViT trunk. It sat latent because the one ConvNeXt arm that ran predates $THROUGHPUT.
    """
    trunk = _Checkpointing()

    TrunkAsTokens.from_sequence(trunk, 4).gradient_checkpointing_enable({"use_reentrant": False})

    assert trunk.asked == [{"use_reentrant": False}], "the trunk is what checkpoints, and the kwargs travel intact"


def test_is_gradient_checkpointing(feature_map: Callable[[int, int], nn.Module]):
    """The READBACK has to reach the trunk too, or the switch lands and reports itself unlanded.

    `KneeModel.applied` reads this off the backbone and `switch_coverage` refuses a run whose switch it
    cannot see. Forwarding the hook without the readback produced exactly that: `recompute_activations:
    asked True, the model shows False` on a model where it had in fact been applied.
    """
    assert not TrunkAsTokens.from_map(feature_map(4, 3), 4).is_gradient_checkpointing, (
        "a trunk that does not report the flag at all reads as off, rather than raising"
    )

    trunk = _Checkpointing()
    adapted = TrunkAsTokens.from_sequence(trunk, 4)

    assert not adapted.is_gradient_checkpointing, "nothing asked of it yet"

    adapted.gradient_checkpointing_enable({"use_reentrant": False})

    assert adapted.is_gradient_checkpointing, "what the trunk says is what the wrapper reports"


def test_from_spatial():
    """The prefix is the mean over TOKENS, not over width — the axis a silent bug would swap."""
    spatial = torch.tensor([[[0.0, 10.0], [2.0, 20.0]]])

    prefixed = TokenOutput.from_spatial(spatial).last_hidden_state

    assert prefixed.shape == (1, 3, 2), "one slot is added, and the width is untouched"
    assert torch.equal(prefixed[0, 0], torch.tensor([1.0, 15.0])), "each channel averaged over its own tokens"
    assert torch.equal(prefixed[0, 1:], spatial[0]), "the spatial tokens survive in their original order"
