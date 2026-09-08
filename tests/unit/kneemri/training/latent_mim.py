from typing import override

import torch
from torch import Tensor, nn

from kneemri.training.latent_mim import FULL_RETENTION, LatentMim, LatentMimConfig

PATCH = 4
GRID = 3
SIDE = PATCH * GRID
TOKENS = GRID * GRID
DIM = 8
BATCH = 2
PREFIX = 1
BLOCKS = 6
CONFIG = LatentMimConfig(patch=PATCH, teacher_blocks=BLOCKS)


class _Backbone(nn.Module):
    """A stand-in for `Dinov2Model` answering BOTH calls this objective makes of it.

    Both, because the student is asked for `last_hidden_state` on a masked view and the teacher for
    `hidden_states` on the whole image, and a stub that only served one would leave the target path — the
    entire subject of this module — unexercised.
    """

    def __init__(self, prefix: int = PREFIX, blocks: int = BLOCKS + 2) -> None:
        super().__init__()
        self.prefix = prefix
        self.blocks = blocks
        self.projection = nn.Linear(SIDE * SIDE, DIM)
        self.scale = nn.Parameter(torch.ones(1))

    def _tokens(self, pixel_values: Tensor) -> Tensor:
        token = self.projection(pixel_values.mean(dim=1).flatten(1)).unsqueeze(1) * self.scale
        return token.expand(-1, self.prefix + TOKENS, DIM).contiguous()

    @override
    def forward(
        self,
        pixel_values: Tensor,
        bool_masked_pos: Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> object:
        tokens = self._tokens(pixel_values)
        if bool_masked_pos is not None:
            # The mask covers the PATCH tokens only, so it is added at the tail — the same end the
            # objective trims the prefix from.
            marked = nn.functional.pad(bool_masked_pos.float(), (self.prefix, 0))
            tokens = tokens + marked.unsqueeze(-1)
        # Each block a distinct multiple, so an average over the WRONG blocks is a different number.
        states = tuple(tokens * float(index + 1) for index in range(self.blocks))
        return type("Output", (), {"last_hidden_state": tokens, "hidden_states": states})


def _images(seed: int = 0) -> Tensor:
    return torch.rand(BATCH, 1, SIDE, SIDE, generator=torch.Generator().manual_seed(seed))


def _objective(config: LatentMimConfig = CONFIG) -> LatentMim:
    return LatentMim(_Backbone(), config)


def _mask() -> Tensor:
    mask = torch.zeros(BATCH, TOKENS, dtype=torch.bool)
    mask[:, :3] = True
    return mask


def test_plan():
    """Momentum is DERIVED from the run's length, so the surviving stock fraction is the declared one.

    This is the whole reason the objective has a `plan` hook at all: a published momentum was fitted for
    a different number of updates from a different init, and at ours it annihilates the prior.
    """
    objective = _objective()
    assert objective.momentum == FULL_RETENTION, "an unplanned run holds the teacher frozen"

    updates = 1000
    objective.plan(updates)
    assert torch.isclose(torch.tensor(objective.momentum**updates), torch.tensor(CONFIG.retention), atol=1e-6)
    assert objective.momentum < FULL_RETENTION

    objective.plan(0)
    assert objective.momentum == CONFIG.retention, "a zero-update run must not divide by zero"


def test_after_step():
    """The teacher moves toward the student only once planned, and not at all while frozen."""
    frozen = _objective()
    before = frozen.teacher.projection.weight.detach().clone()
    frozen.backbone.projection.weight.data.add_(1.0)
    frozen.after_step()
    assert torch.equal(frozen.teacher.projection.weight, before), "retention 1.0 means a frozen teacher"

    moving = _objective()
    moving.plan(2)
    start = moving.teacher.projection.weight.detach().clone()
    moving.backbone.projection.weight.data.add_(1.0)
    moving.after_step()
    moved = moving.teacher.projection.weight
    assert not torch.equal(moved, start), "a planned teacher must track"
    assert torch.all((moved - start).abs() < 1.0), "it must track SLOWLY, not copy"


def test_target():
    """The teacher reads the WHOLE image, its top K blocks are normalised each, and the prefix is dropped.

    Layer-normalising per block before averaging is data2vec's explicit anti-collapse device, and it is
    also what stops the largest-norm block from being the target by construction. Both are asserted here
    because an unnormalised average has the same shape and a different meaning.
    """
    objective = _objective()
    target = objective.target(_images(), TOKENS)
    assert target.shape == (BATCH, TOKENS, DIM)

    # A block scaled by a constant is unchanged after parameter-less layer norm, so scaling the teacher's
    # output must not move the target at all — which is precisely the domination the normalisation kills.
    objective.teacher.scale.data.mul_(100.0)
    assert torch.allclose(objective.target(_images(), TOKENS), target, atol=1.0e-4)

    fewer = _objective(LatentMimConfig(patch=PATCH, teacher_blocks=1))
    assert not torch.allclose(fewer.target(_images(), TOKENS), target), "K must select which blocks are averaged"


def test_latent_loss():
    """Zero against the target itself, and blind to whatever is predicted at the VISIBLE positions."""
    objective = _objective()
    target = torch.rand(BATCH, TOKENS, DIM)
    mask = _mask()
    assert objective.latent_loss(target, target, mask) == 0.0

    wrong_visible = target.clone()
    wrong_visible[:, 3:] += 10.0
    assert objective.latent_loss(wrong_visible, target, mask) == 0.0, "visible positions must not be scored"

    wrong_hidden = target.clone()
    wrong_hidden[:, :3] += 10.0
    assert objective.latent_loss(wrong_hidden, target, mask) > 0.0


def test_spread():
    """The collapse alarm: it falls to zero exactly when every masked target is the same vector.

    That failure drives the loss to zero and its curve is indistinguishable from success, so this is the
    only number that tells the two apart.
    """
    objective = _objective()
    mask = _mask()
    collapsed = torch.ones(BATCH, TOKENS, DIM)
    assert objective.spread(collapsed, mask) == 0.0

    varied = torch.rand(BATCH, TOKENS, DIM, generator=torch.Generator().manual_seed(3))
    assert objective.spread(varied, mask) > 0.0


def test_term_names():
    """The loss and the alarm are named, because the alarm is what a flat curve has to be read against."""
    assert _objective().term_names() == ("latent", "spread")


def test_forward():
    """The loss, itself, and the alarm beside it — and the loss must actually depend on the student."""
    objective = _objective()
    images = _images()
    mask = _mask()
    total, latent, spread = objective.forward(images, mask)
    assert torch.equal(total, latent)
    assert torch.isfinite(spread)
    assert latent > 0.0, "an untrained student on a masked view cannot already match its teacher"
    assert latent.requires_grad, "the loss must carry a gradient back into the student"


def test_config_defaults_are_the_ones_the_write_up_will_quote():
    """The three numbers this objective is free to move, pinned against a silent default change."""
    config = LatentMimConfig()
    assert (config.retention, config.teacher_blocks, config.smooth_beta) == (0.6, 6, 2.0)
