"""The forward noising a diffusion trunk's features are read AT, written out rather than imported.

A diffusion model is not an encoder until a timestep is chosen: its activations are a function of a
NOISED volume, so `q_sample` is part of the feature definition and not a training detail. OrthoDiffusion
reads features at a fixed small `t`, which is what makes the trunk deterministic enough to cache.

WHY THIS IS OURS AND NOT THEIRS. Their `q_sample` lives in `diffusion_model/trainer.py`, which imports
einops, nibabel, tensorboard and torchio — none of which this project has, and none of which the six
lines below need. The schedule itself is thirty lines of arithmetic (`cosine_beta_schedule`, betas ->
alphas -> cumulative product -> two square roots), published in their file and in the paper it cites, so
writing it here costs a dependency chain and buys an exact reimplementation that a test can pin.

THE NOISE IS SEEDED, DELIBERATELY. `torch.randn_like` inside their sampler makes every encode a fresh
draw, which for a cached feature array means the cache silently describes one arbitrary sample. A
generator argument turns that into a stated choice: the same study encodes to the same vector twice.
"""

import math

import torch
from jaxtyping import Float
from torch import Tensor

TIMESTEPS = 1000
"""The schedule length the published weights were trained under. Not a free choice — the trunk's time
embedding was fitted against these indices, so a different length reads the wrong point of the curve."""

COSINE_OFFSET = 0.008
"""Their `s`, the small offset that keeps the schedule from being singular at t=0. Nichol & Dhariwal."""

BETA_CEILING = 0.999
"""Their clip. A beta of exactly 1 destroys the signal entirely and makes the posterior undefined."""


class DiffusionSchedule:
    """The cosine variance schedule, and the one operation a frozen trunk needs from it."""

    def __init__(self, timesteps: int = TIMESTEPS, offset: float = COSINE_OFFSET) -> None:
        self.timesteps = timesteps
        self.offset = offset
        alphas = self._alphas_cumulative()
        self.sqrt_alphas = alphas.sqrt()
        self.sqrt_one_minus_alphas = (1.0 - alphas).sqrt()

    def _alphas_cumulative(self) -> Float[Tensor, " timesteps"]:
        """Cumulative alpha at every step, by way of the betas their schedule clips.

        Built through the betas rather than read off the cosine directly, because the clip at
        `BETA_CEILING` is not a no-op: it bends the tail, and a version that skipped it would agree with
        theirs everywhere except the last few steps and disagree silently there.
        """
        steps = self.timesteps + 1
        grid = torch.linspace(0, steps, steps, dtype=torch.float64)
        curve = torch.cos(((grid / steps) + self.offset) / (1.0 + self.offset) * math.pi * 0.5) ** 2
        curve = curve / curve[0]
        betas = torch.clamp(1.0 - (curve[1:] / curve[:-1]), min=0.0, max=BETA_CEILING)
        return torch.cumprod(1.0 - betas, dim=0)

    def noised(
        self,
        volume: Float[Tensor, "b c d h w"],
        timestep: int,
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "b c d h w"]:
        """One forward-diffusion step: the volume scaled down, plus noise scaled up.

        The scalars are broadcast rather than gathered per-sample because a frozen read uses ONE
        timestep for the whole batch — their `extract` exists to support a per-sample `t` that only
        training needs.
        """
        noise = torch.randn(volume.shape, generator=generator, dtype=torch.float32, device=volume.device)
        signal = float(self.sqrt_alphas[timestep])
        residual = float(self.sqrt_one_minus_alphas[timestep])
        return volume * signal + noise.to(volume.dtype) * residual
