"""The perturbations, tested on pixels alone — no model, no optimiser, no card."""

import torch

from kneemri.training.augmentation import Augmentation, AugmentConfig

SEED = 2026
SLOTS = 6


def test_applied():
    """Jitter moves the pixels but never mirrors them — a flip would undo laterality normalisation."""
    generator = torch.Generator().manual_seed(SEED)
    images = torch.rand(4, 3, 16, 16)

    warped = Augmentation().applied(images, generator)

    assert warped.shape == images.shape
    assert not torch.allclose(warped, images), "augmentation must actually perturb"
    assert float(warped.max()) <= 1.0
    flipped = torch.flip(images, dims=[-1])
    assert not torch.allclose(warped, flipped, atol=0.2), "no horizontal flip may be introduced"


def test_bagged():
    """Bags stay uint8 and keep their shape, and every slot is perturbed rather than passed through."""
    torch.manual_seed(SEED)
    bag = torch.randint(0, 256, (2, SLOTS, 3, 16, 16), dtype=torch.uint8)

    out = Augmentation().bagged(bag, torch.Generator().manual_seed(SEED))

    assert out.shape == bag.shape
    assert out.dtype == torch.uint8
    assert not torch.equal(out, bag), "augmentation must actually perturb the bag"


def test_geometric_only():
    """The default must reproduce the pre-existing recipe exactly, or it silently changes members.

    Folds trained before the intensity block existed have to remain comparable with folds trained
    after it, so `geometric_only` is asserted to leave the intensity distribution alone: every pixel
    of an image scales by the SAME factor, which is what a gain is and a gamma/contrast/blur is not.
    """
    images = torch.rand(4, 3, 16, 16) * 0.8 + 0.1

    out = Augmentation(AugmentConfig.geometric_only())._intensity(images, torch.Generator().manual_seed(7))

    ratio = out / images
    assert float(ratio.flatten(1).std(1).max()) < 1e-3, "a pure gain scales every pixel identically"


def test_geometric_only_does_not_compute_the_ops_it_disables():
    """A disabled transform is SKIPPED, not computed and multiplied by zero.

    Masked arithmetic gives the right answer either way, which is why this went unnoticed: every step
    of every run in the ledger paid for a full-tensor `pow` and a `conv2d` blur whose outputs were
    discarded — 15.5 ms and 3.23 GB of transient allocation a step against 2.0 ms and 1.29 GB, at
    the whole-stack shape on a 5090.

    Asserted through the GENERATOR, which is the only externally visible trace an op leaves when its
    result is thrown away: gamma, blur and contrast each draw from it, so a config that disables all
    three must advance it by exactly the gain's one draw. Comparing outputs cannot see this, and
    timing it would be a flaky test.
    """
    images = torch.rand(4, 3, 16, 16)

    gain_only = torch.Generator().manual_seed(7)
    torch.rand((4, 1, 1, 1), generator=gain_only)
    spent = torch.Generator().manual_seed(7)
    Augmentation(AugmentConfig.geometric_only())._intensity(images, spent)

    assert torch.equal(spent.get_state(), gain_only.get_state()), "a disabled op still drew from the generator"


def test_intensity_perturbs_and_stays_in_range():
    """The full config must actually bite, and must not leave [0, 1] — the cache is uint8."""
    images = torch.rand(8, 3, 16, 16)

    out = Augmentation(AugmentConfig())._intensity(images, torch.Generator().manual_seed(7))

    assert float((out - images).abs().gt(1e-4).float().mean()) > 0.5, "intensity augmentation must perturb"
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
