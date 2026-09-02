"""The mirror for the cosine schedule: the arithmetic pinned against the properties it must have."""

import torch

from kneemri.models.diffusion_schedule import DiffusionSchedule

STEPS = 200
"""A short schedule, so the test builds its own curve rather than the trunk's thousand-step one."""

TOLERANCE = 1e-3
"""How far the first cumulative alpha may sit below 1: it is `1 - beta_0`, and beta_0 grows as the
schedule shortens, so a tolerance tuned on the trunk's thousand steps would fail on this test's two
hundred for a reason that is about the fixture."""


def test_noised() -> None:
    """A seeded draw is reproducible, and an early timestep leaves the signal dominant."""
    schedule = DiffusionSchedule(timesteps=STEPS)
    volume = torch.ones((2, 1, 4, 8, 8))
    first = schedule.noised(volume, 1, torch.Generator().manual_seed(0))
    again = schedule.noised(volume, 1, torch.Generator().manual_seed(0))
    assert torch.allclose(first, again)
    assert first.shape == volume.shape
    late = schedule.noised(volume, STEPS - 1, torch.Generator().manual_seed(0))
    assert first.std() < late.std()


def test_alphas_fall_monotonically() -> None:
    """The cumulative alphas start at ~1 and decay, which is what makes a timestep a noise level."""
    schedule = DiffusionSchedule(timesteps=STEPS)
    assert schedule.sqrt_alphas[0] > 1.0 - TOLERANCE
    assert bool(torch.all(schedule.sqrt_alphas[1:] <= schedule.sqrt_alphas[:-1]))
    assert schedule.sqrt_alphas[-1] < schedule.sqrt_one_minus_alphas[-1]
