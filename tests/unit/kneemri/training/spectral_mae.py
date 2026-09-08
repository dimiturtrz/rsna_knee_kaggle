from typing import override

import numpy as np
import torch
from torch import nn

from kneemri.training.spectral_mae import SpectralMae, SpectralMaskConfig

PATCH = 4
GRID = 3
SIDE = PATCH * GRID
TOKENS = GRID * GRID
DIM = 8
BATCH = 2
PREFIX = 1
CONFIG = SpectralMaskConfig(patch=PATCH)


class _Backbone(nn.Module):
    """A stand-in for `Dinov2Model` that records what it was masked with.

    Written rather than downloaded because a unit test may not reach the network, and because the two
    facts worth pinning about the encoder call — that the mask reaches it, and that the prefix token is
    dropped from what comes back — are facts about OUR code, not about DINOv2.
    """

    def __init__(self, prefix: int = PREFIX) -> None:
        super().__init__()
        self.prefix = prefix
        self.projection = nn.Linear(SIDE * SIDE, DIM)
        self.seen: torch.Tensor | None = None

    @override
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> object:
        self.seen = bool_masked_pos
        flat = pixel_values.mean(dim=1).flatten(1)
        token = self.projection(flat).unsqueeze(1)
        return type("Output", (), {"last_hidden_state": token.expand(-1, self.prefix + TOKENS, DIM).contiguous()})


def _images(seed: int = 0) -> torch.Tensor:
    return torch.rand(BATCH, 1, SIDE, SIDE, generator=torch.Generator().manual_seed(seed))


def _objective() -> SpectralMae:
    return SpectralMae(_Backbone(), DIM, CONFIG)


def test_patchify():
    """Rows must be patches in raster order, so the decoder predicts the patch it was handed.

    Pinned against a hand-built image whose every patch is a distinct constant: a transposed or
    column-major reshape gives the same SHAPE and a different row order, which no downstream assertion
    about sizes could ever catch.
    """
    marked = torch.arange(TOKENS, dtype=torch.float32).repeat_interleave(PATCH * PATCH)
    image = marked.reshape(GRID, GRID, PATCH, PATCH).permute(0, 2, 1, 3).reshape(1, 1, SIDE, SIDE)
    rows = _objective().patchify(image)
    assert rows.shape == (1, TOKENS, PATCH * PATCH)
    assert rows.std(dim=-1).max() == 0.0, "each patch was a constant, so no row may mix two of them"
    assert torch.equal(rows[0, :, 0], torch.arange(TOKENS, dtype=torch.float32)), "raster order, row-major"


def test_unpatchify():
    """The exact inverse of `patchify`, which is what lets the spectral term see anatomy and not tiles."""
    images = _images()
    objective = _objective()
    assert torch.allclose(objective.unpatchify(objective.patchify(images), GRID), images)


def test_sample_mask():
    """Every sample hides the SAME number of patches, at independently drawn positions."""
    objective = _objective()
    generator = torch.Generator().manual_seed(1)
    mask = objective.sample_mask(BATCH, TOKENS, torch.device("cpu"), generator)
    assert mask.shape == (BATCH, TOKENS)
    assert mask.dtype == torch.bool
    hidden = round(TOKENS * CONFIG.mask_ratio)
    assert mask.sum(dim=1).tolist() == [hidden, hidden], "a per-sample count would reweight the batch mean"

    # The seed alone must determine the masking, which is what a CPU generator buys and a device-side one
    # would lose. A CUDA run cannot be asserted here, and the smoke run is what covers that half.
    repeat = objective.sample_mask(BATCH, TOKENS, torch.device("cpu"), torch.Generator().manual_seed(1))
    assert torch.equal(repeat, mask), "the same seed must give the same mask, on any device"

    ratios = [SpectralMae(_Backbone(), DIM, SpectralMaskConfig(patch=PATCH, mask_ratio=r)) for r in (0.0, 1.0)]
    counts = [m.sample_mask(1, TOKENS, torch.device("cpu"), generator).sum().item() for m in ratios]
    assert counts == [1, TOKENS], "the boundaries: never fewer than one hidden patch, never more than all"


def test_encode():
    """The mask reaches the backbone, and the prefix tokens are dropped from what comes back.

    Both halves matter and neither is visible in a shape alone at prefix 1. The DINOv3 case — four
    register tokens — is the one that would silently misalign every patch by four positions, so the
    prefix width is varied here rather than assumed.
    """
    objective = _objective()
    mask = torch.zeros(BATCH, TOKENS, dtype=torch.bool)
    mask[:, 0] = True
    tokens = objective.encode(_images(), mask)
    assert tokens.shape == (BATCH, TOKENS, DIM)
    assert objective.backbone.seen is not None, "the backbone was never told what to mask"
    assert torch.equal(objective.backbone.seen, mask)

    registers = SpectralMae(_Backbone(prefix=5), DIM, CONFIG)
    assert registers.encode(_images(), mask).shape == (BATCH, TOKENS, DIM), "counted from the wrong end"


def test_restore():
    """A perfect standardised prediction comes back as the target itself, in the target's own units."""
    objective = _objective()
    target = torch.rand(BATCH, TOKENS, PATCH * PATCH)

    centred = target - target.mean(dim=-1, keepdim=True)
    exact = centred / centred.var(dim=-1, keepdim=True).sqrt()
    recovered = objective.restore(exact, target)
    assert torch.allclose(recovered, target, atol=1.0e-4), "the inverse of the pixel loss's standardisation"

    flat = torch.zeros(1, 1, PATCH * PATCH)
    assert torch.isfinite(objective.restore(flat, flat)).all(), "a background patch has variance zero"


def test_pixel_loss():
    """Only the hidden patches are scored, and a flat patch does not divide by its own zero variance."""
    objective = _objective()
    target = torch.rand(BATCH, TOKENS, PATCH * PATCH)
    mask = torch.zeros(BATCH, TOKENS, dtype=torch.bool)
    mask[:, 0] = True

    centred = target - target.mean(dim=-1, keepdim=True)
    exact = centred / centred.var(dim=-1, keepdim=True).sqrt()
    assert objective.pixel_loss(exact, target, mask) < 1.0e-4, "the standardised target is the zero of this loss"

    wrong_visible = exact.clone()
    wrong_visible[:, 1:] += 10.0
    assert objective.pixel_loss(wrong_visible, target, mask) < 1.0e-4, "visible patches must not be scored"

    flat = torch.zeros(1, 1, PATCH * PATCH)
    hidden = torch.ones(1, 1, dtype=torch.bool)
    assert torch.isfinite(objective.pixel_loss(flat, flat, hidden)), "a background patch has variance zero"


def test_highpass():
    """A pure DC image has no content above the cutoff; a checkerboard is all content above it."""
    objective = _objective()
    flat = torch.full((1, 1, SIDE, SIDE), 0.5)
    assert objective.highpass(flat).abs().max() < 1.0e-5, "constant intensity is exactly the DC term"

    checker = torch.zeros(1, 1, SIDE, SIDE)
    checker[..., ::2, ::2] = 1.0
    assert objective.highpass(checker).abs().max() > 0.0, "the highest frequency the grid holds must survive"


def test_spectral_loss():
    """Zero against itself, and larger for a blurred reconstruction than for a faithful one.

    Blur is the failure mode the term exists to price: it is nearly free under a pixel MSE and it is
    exactly the loss of the focal detail our vision-bottlenecked labels depend on.
    """
    objective = _objective()
    images = _images()
    assert objective.spectral_loss(images, images) == 0.0

    blurred = nn.functional.avg_pool2d(images, PATCH, stride=1, padding=PATCH // 2)[..., :SIDE, :SIDE]
    assert objective.spectral_loss(blurred, images) > objective.spectral_loss(images + 1.0e-3, images)


def test_term_names():
    """The spectral term IS this module's claim, so the log line has to name it rather than a total."""
    assert _objective().term_names() == ("pixel", "spectral")


def test_forward():
    """Both terms come back separately, and the weight actually multiplies the spectral one.

    The separation is not cosmetic: a total that improves while the spectral term is flat means the run
    is a plain masked autoencoder under this module's name, and only a returned pair can show that.
    """
    objective = _objective()
    images = _images()
    mask = objective.sample_mask(BATCH, TOKENS, torch.device("cpu"), torch.Generator().manual_seed(2))
    total, pixels, spectral = objective.forward(images, mask)
    assert torch.allclose(total, pixels + CONFIG.spectral_weight * spectral)
    assert spectral > 0.0, "an untrained decoder cannot already match the spectrum"

    doubled = SpectralMae(objective.backbone, DIM, SpectralMaskConfig(patch=PATCH, spectral_weight=2.0))
    doubled.decoder = objective.decoder
    assert doubled(images, mask)[0] > total, "the weight must reach the sum"


def test_encoder_state():
    """The encoder alone, keyed as the stock model keys it — no decoder, no normalisation buffers."""
    state = _objective().encoder_state()
    assert state, "an empty state dict is the silent failure this whole module is written against"
    assert not [name for name in state if name.startswith(("decoder", "mean", "std"))]
    assert all(value.device.type == "cpu" for value in state.values())


def test_config_defaults_are_the_ones_the_write_up_will_quote():
    """The three numbers an arm is free to move, pinned so a silent default change is a failing test."""
    assert (CONFIG.mask_ratio, CONFIG.spectral_weight, CONFIG.highpass) == (0.6, 1.0, 0.125)
    assert np.isclose(SpectralMaskConfig().patch, 14), "DINOv2 tiles at 14; DINOv3 must pass its own"
