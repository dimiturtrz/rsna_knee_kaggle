from typing import override

import pytest
import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from kneemri.training.masked_objective import MaskConfig, MaskedObjective

PATCH = 4
GRID = 3
SIDE = PATCH * GRID
TOKENS = GRID * GRID
DIM = 8
BATCH = 2
PREFIX = 1
CONFIG = MaskConfig(patch=PATCH)


class _Backbone(nn.Module):
    """A stand-in for `Dinov2Model` that records what it was masked with.

    Written rather than downloaded because a unit test may not reach the network, and because the facts
    worth pinning here — that the mask reaches the encoder and that the prefix is dropped from what comes
    back — are facts about OUR code, not about DINOv2.
    """

    def __init__(self, prefix: int = PREFIX) -> None:
        super().__init__()
        self.prefix = prefix
        self.projection = nn.Linear(SIDE * SIDE, DIM)
        self.seen: Tensor | None = None

    @override
    def forward(self, pixel_values: Tensor, bool_masked_pos: Tensor | None = None) -> object:
        self.seen = bool_masked_pos
        token = self.projection(pixel_values.mean(dim=1).flatten(1)).unsqueeze(1)
        return type("Output", (), {"last_hidden_state": token.expand(-1, self.prefix + TOKENS, DIM).contiguous()})


class _Stub(MaskedObjective[MaskConfig]):
    """The smallest concrete objective there is: the base plus the two abstract members, and nothing else.

    A concrete subclass rather than the base itself because `MaskedObjective` cannot be instantiated, and
    the point of these tests is the shared half — masking, normalisation, prefix trimming, what is saved.
    """

    @override
    def term_names(self) -> tuple[str, str]:
        return ("first", "second")

    @override
    def forward(
        self, images: Float[Tensor, "b 1 h w"], mask: Bool[Tensor, "b n"]
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
        encoded = self.encode(images, mask)
        first = encoded.mean()
        return first, first, encoded.std()


def _images(seed: int = 0) -> Float[Tensor, "b 1 h w"]:
    return torch.rand(BATCH, 1, SIDE, SIDE, generator=torch.Generator().manual_seed(seed))


def _objective(prefix: int = PREFIX) -> _Stub:
    return _Stub(_Backbone(prefix), CONFIG)


def test_config():
    """The knobs come back as they went in, and `config` is READ-ONLY.

    Read-only is the whole reason it is a property rather than an attribute: a mutable one cannot be
    narrowed by a subclass, which is what forces every call site to spell `MaskedObjective[MaskConfig]`.
    """
    objective = _objective()
    assert objective.config is CONFIG
    with pytest.raises(AttributeError):
        objective.config = MaskConfig(patch=PATCH)


def test_sample_mask():
    """Every sample hides the SAME number of patches, at independently drawn positions."""
    objective = _objective()
    generator = torch.Generator().manual_seed(1)
    mask = objective.sample_mask(BATCH, TOKENS, torch.device("cpu"), generator)
    assert mask.shape == (BATCH, TOKENS)
    assert mask.dtype == torch.bool
    hidden = round(TOKENS * CONFIG.mask_ratio)
    assert mask.sum(dim=1).tolist() == [hidden, hidden], "a per-sample count would reweight the batch mean"

    repeat = objective.sample_mask(BATCH, TOKENS, torch.device("cpu"), torch.Generator().manual_seed(1))
    assert torch.equal(repeat, mask), "the seed alone must determine the masking, on any device"

    edges = [_Stub(_Backbone(), MaskConfig(patch=PATCH, mask_ratio=ratio)) for ratio in (0.0, 1.0)]
    counts = [edge.sample_mask(1, TOKENS, torch.device("cpu"), generator).sum().item() for edge in edges]
    assert counts == [1, TOKENS], "the boundaries: never fewer than one hidden patch, never more than all"


def test_normalise():
    """One cached slice becomes three channels in the TRUNK'S OWN units, not somebody else's.

    The channel replication and the statistics are asserted together because either alone passes a shape
    check: pretraining a checkpoint through the wrong normalisation is a silent handicap on the one thing
    the whole run is about.
    """
    objective = _objective()
    images = _images()
    normalised = objective.normalise(images)
    assert normalised.shape == (BATCH, 3, SIDE, SIDE)
    expected = (images.expand(-1, 3, -1, -1) - objective.mean) / objective.std
    assert torch.allclose(normalised, expected)


def test_patches():
    """The prefix is dropped by counting BACK, so four registers do not shift every patch by four.

    Counted back from the end because only the patch count is known at this layer. A forward index passes
    at prefix 1 and misaligns DINOv3 silently, which is exactly the bug no shape assertion can see.
    """
    objective = _objective()
    hidden = torch.arange(5 + TOKENS, dtype=torch.float32).view(1, 5 + TOKENS, 1).expand(1, 5 + TOKENS, DIM)
    kept = objective.patches(hidden.contiguous(), TOKENS)
    assert kept.shape == (1, TOKENS, DIM)
    assert kept[0, 0, 0] == 5.0, "the five prefix tokens, and only those, were dropped"


def test_encode():
    """The mask reaches the backbone, and the prefix tokens are dropped from what comes back."""
    objective = _objective()
    mask = torch.zeros(BATCH, TOKENS, dtype=torch.bool)
    mask[:, 0] = True
    assert objective.encode(_images(), mask).shape == (BATCH, TOKENS, DIM)
    assert objective.backbone.seen is not None, "the backbone was never told what to mask"
    assert torch.equal(objective.backbone.seen, mask)

    registers = _objective(prefix=5)
    assert registers.encode(_images(), mask).shape == (BATCH, TOKENS, DIM), "counted from the wrong end"


def test_plan():
    """The default objective needs nothing from the run's length, and must not break on being told it."""
    objective = _objective()
    assert objective.plan(1000) is None


def test_after_step():
    """The default objective owes the optimiser step nothing, and must not break on being asked."""
    objective = _objective()
    assert objective.after_step() is None


def test_term_names():
    """An objective names its own two reported numbers, so a log line is readable without the source."""
    first, second = _objective().term_names()
    assert (first, second) == ("first", "second")


def test_forward():
    """The total and both reported terms come back, in the order `term_names` names them."""
    objective = _objective()
    mask = objective.sample_mask(BATCH, TOKENS, torch.device("cpu"), torch.Generator().manual_seed(2))
    total, first, second = objective.forward(_images(), mask)
    assert total.shape == ()
    assert torch.equal(total, first)
    assert torch.isfinite(second)


def test_encoder_state():
    """The encoder alone, keyed as the stock model keys it — no normalisation buffers, on the CPU."""
    state = _objective().encoder_state()
    assert state, "an empty state dict is the silent failure this whole module is written against"
    assert not [name for name in state if name.startswith(("mean", "std"))]
    assert all(value.device.type == "cpu" for value in state.values())


def test_config_defaults_are_the_ones_the_write_up_will_quote():
    """The two numbers deciding what an objective is SHOWN, pinned against a silent default change."""
    assert (MaskConfig().mask_ratio, MaskConfig().patch) == (0.6, 14)
