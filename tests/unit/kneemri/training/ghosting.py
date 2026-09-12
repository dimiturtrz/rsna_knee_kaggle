"""What the motion-ghosting operator must do, including the one thing it must NOT do at its default."""

import torch

from kneemri.training.ghosting import DISPLACEMENTS, Ghosting


def _images(count: int = 8, size: int = 32) -> torch.Tensor:
    """A batch with structure to displace: a bright block on a dark field, which is what a crop looks like."""
    images = torch.zeros(count, 1, size, size)
    images[:, :, size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 0.8
    return images


def test_severity() -> None:
    assert Ghosting(0.04).severity == 0.04


def test_targets() -> None:
    """Lognormal around the median: strictly positive, and its own median back at the severity."""
    drawn = Ghosting(0.04).targets(4096, torch.Generator().manual_seed(0), torch.device("cpu"))
    assert drawn.shape == (4096,)
    assert bool((drawn > 0).all())
    assert abs(float(drawn.median()) - 0.04) < 0.004


def test_applied() -> None:
    """The operator lands on the residual it was asked for, and leaves the image in range."""
    images = _images()
    out = Ghosting(0.04).applied(images, torch.Generator().manual_seed(0))
    assert out.shape == images.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    residual = float((out - images).abs().mean())
    assert 0.02 < residual < 0.08


def test_applied_is_identity_at_zero() -> None:
    """Zero severity returns the pixels untouched — the recipe every member on disk was fitted under."""
    images = _images()
    assert torch.equal(Ghosting(0.0).applied(images, torch.Generator().manual_seed(0)), images)


def test_applied_is_impulsive() -> None:
    """The residual concentrates on edges rather than spreading — the property KMAR measures at ~9.

    This is the check that refuted the operator's first build, which displaced whole-object replicas by
    a fraction of the frame and came back at 3.39. The bar here is 5.0: well above additive noise (~2.6),
    well below the corpus, and chosen so the test asserts the SHAPE the mechanism must have rather than
    a number measured on a different corpus at a different crop.
    """
    images = _images()
    difference = (Ghosting(0.04).applied(images, torch.Generator().manual_seed(0)) - images).abs()
    assert float(difference.quantile(0.99)) / float(difference.mean()) > 5.0


def test_applied_displaces_along_one_axis() -> None:
    """A displacement small enough that the bright block still overlaps itself after the shift."""
    assert max(DISPLACEMENTS) < 8
