"""The perturbations, tested on pixels alone — no model, no optimiser, no card."""

import pytest
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


def _study_bag(studies: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """A bag whose LAST slot is absent for every study, which is the case the rate must never count."""
    torch.manual_seed(SEED)
    images = torch.randint(1, 256, (studies, SLOTS, 3, 8, 8), dtype=torch.uint8)
    presence = torch.ones(studies, SLOTS)
    presence[:, -1] = 0.0
    images[:, -1] = 0
    return images, presence


def test_dropped():
    """Pixels and the presence mask move TOGETHER, and the identity rate is a genuine pass-through."""
    images, presence = _study_bag()
    generator = torch.Generator().manual_seed(SEED)

    kept_images, kept_presence = Augmentation(AugmentConfig(slot_dropout=0.5)).dropped(images, presence, generator)

    assert kept_images.shape == images.shape
    assert kept_presence.shape == presence.shape
    dropped = kept_presence == 0
    assert bool(dropped[:, :-1].any()), "a rate of 0.5 over eight studies must withhold something"
    assert float(kept_images[dropped].abs().sum()) == 0.0, (
        "a slot cleared from the mask must carry no pixels — a zeroed slot still called present "
        "teaches the head that a present slot may be empty"
    )
    survived = ~dropped
    assert torch.equal(kept_images[survived], images[survived]), "a kept slot is untouched"

    off_images, off_presence = Augmentation(AugmentConfig()).dropped(images, presence, generator)
    assert torch.equal(off_images, images), "the default rate is off and must not copy or perturb"
    assert torch.equal(off_presence, presence)


def test_dropped_never_empties_a_study():
    """A draw that would withhold every acquired slot is CANCELLED, not clamped to one survivor."""
    images, presence = _study_bag(studies=64)

    _, kept = Augmentation(AugmentConfig(slot_dropout=0.99)).dropped(
        images, presence, torch.Generator().manual_seed(SEED)
    )

    assert int(kept.sum(dim=1).min()) >= 1, "no study may reach the head with nothing acquired"


def test_dropped_never_selects_an_absent_slot():
    """The rate is a rate over what was ACQUIRED, so an absent slot cannot be spent on."""
    images, presence = _study_bag(studies=64)

    _, kept = Augmentation(AugmentConfig(slot_dropout=1.0)).dropped(
        images, presence, torch.Generator().manual_seed(SEED)
    )

    assert float(kept[:, -1].sum()) == 0.0, "an absent slot stays absent"
    assert int(kept[:, :-1].sum(dim=1).min()) == SLOTS - 1, (
        "at rate 1.0 every acquired slot is drawn, the study would empty, and cancelling that draw "
        "restores the whole acquired set — the degenerate rate is an identity, not a wipe"
    )


def test_augment_config_intensity():
    """The block's presence is read off the WIDTHS, since `geometric_only` disables it by zeroing them."""
    assert AugmentConfig().intensity() is True
    assert AugmentConfig.geometric_only().intensity() is False
    assert AugmentConfig(gamma_probability=0.0, blur_probability=0.0, noise=0.0).intensity() is True, (
        "a contrast range wider than a point is still an intensity perturbation"
    )


def test_augmentation_intensity():
    """A trainer asks its augmentation, not the config it thinks it passed — the default is geometric."""
    assert Augmentation().intensity() is False
    assert Augmentation(AugmentConfig()).intensity() is True


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


def test_scaled():
    """Full strength is the config itself; zero is the identity op; half halves every width about its identity."""
    config = AugmentConfig(slot_dropout=0.4, ghosting=0.04)
    assert config.scaled(1.0) is config, "the default schedule must hand back the ledger's config untouched"
    none = config.scaled(0.0)
    assert (none.rotation_degrees, none.scale_jitter, none.noise, none.slot_dropout, none.ghosting) == (0, 0, 0, 0, 0)
    assert none.gamma == (1.0, 1.0)
    assert none.contrast == (1.0, 1.0)
    half = config.scaled(0.5)
    assert half.slot_dropout == pytest.approx(0.2)
    assert half.gamma[0] == pytest.approx(1.0 + (config.gamma[0] - 1.0) / 2)
    assert config.scaled(10.0).slot_dropout == 1.0, "a probability clips at one"
    assert config.scaled(0.5).padding_mode == config.padding_mode


def test_strengthen():
    """The applied config follows the strength; the asked-for `base` never moves."""
    augmentation = Augmentation(AugmentConfig(slot_dropout=0.4))
    augmentation.strengthen(0.5)
    assert augmentation.config.slot_dropout == pytest.approx(0.2)
    assert augmentation.base.slot_dropout == pytest.approx(0.4)
    augmentation.strengthen(1.0)
    assert augmentation.config is augmentation.base
