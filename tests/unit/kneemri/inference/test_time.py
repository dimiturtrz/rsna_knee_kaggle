"""Deterministic test-time views: the identity must be exact, and a rotation must actually change."""

import pytest
import torch

from core.findings import TARGETS
from kneemri.inference import test_time
from kneemri.inference.test_time import SpatialViews, WindowPooling

SEED = 2026


def test_apply():
    """The identity view is the input itself; a rotated view differs but keeps shape and dtype.

    Identity is returned UNTOUCHED rather than warped by zero degrees: a zero-degree grid_sample still
    resamples, so it would quietly blur every prediction we have ever made the moment views were
    enabled, and the single-view path must be a genuine no-op.
    """
    torch.manual_seed(SEED)
    images = torch.randint(0, 256, (2, 6, 4, 16, 16), dtype=torch.uint8)
    views = list(SpatialViews(degrees=(0.0, 6.0), shifts=((0.0, 0.0),)).apply(images))
    assert len(views) == 2
    assert torch.equal(views[0], images), "the identity view must be the input, not a resampling of it"
    assert views[1].shape == images.shape and views[1].dtype == torch.uint8
    assert not torch.equal(views[1], images), "a six-degree rotation has to change the pixels"


def test_len():
    """The view count is the multiplier on inference cost, so it is stated rather than discovered."""
    assert len(SpatialViews(degrees=(0.0, -6.0, 6.0), shifts=((0.0, 0.0),))) == 3
    assert len(SpatialViews(degrees=(0.0,), shifts=((0.0, 0.0), (0.02, 0.0)))) == 2


def test_warp_is_indifferent_to_the_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunking the warp is a memory fix and must not be an arithmetic one.

    The whole-batch warp allocated its float copy, its sampling grid and its result at the batch's own
    length — 6.2 GB for the 3,456 images a 48-study chunk holds, which is an out-of-memory the forward's
    row-halving retry never sees because it is raised before the forward (knee-c9q4). The fix is only
    legitimate if every view is bit-identical to the one the unchunked path produced, which is what this
    asserts: same pixels at a chunk that splits the batch as at one that cannot.
    """
    torch.manual_seed(SEED)
    images = torch.randint(0, 256, (3, 6, 4, 12, 12), dtype=torch.uint8)
    views = SpatialViews(degrees=(-6.0,), shifts=((0.0, 0.0),))

    monkeypatch.setattr(test_time, "WARP_CHUNK", 10_000)
    whole = next(iter(views.apply(images)))
    monkeypatch.setattr(test_time, "WARP_CHUNK", 7)
    pieces = next(iter(views.apply(images)))

    assert whole.shape == images.shape
    assert torch.equal(whole, pieces), "a chunk boundary changed a pixel, so the fix moved the answer"


def test_spatial_views_identity() -> None:
    """One unwarped view is the plain model; one ROTATED view is not, and length cannot tell them apart.

    Whoever asks whether an array computed elsewhere may stand in for this reading needs that
    distinction: `SpatialViews(degrees=(6.0,))` is also a single forward pass and a different function.
    """
    assert SpatialViews(degrees=(0.0,), shifts=((0.0, 0.0),)).identity
    assert not SpatialViews(degrees=(6.0,), shifts=((0.0, 0.0),)).identity
    assert not SpatialViews().identity


def test_uniform() -> None:
    """One reducer named for all twelve targets, and a name the reducers do not know is refused."""
    assert WindowPooling.uniform("max").per_target == tuple("max" for _ in TARGETS)
    assert WindowPooling.uniform("mean") == WindowPooling()
    with pytest.raises(ValueError, match="unknown window reducers"):
        WindowPooling.uniform("median")
    with pytest.raises(ValueError, match="reducers for"):
        WindowPooling(per_target=("mean", "max"))


def test_focal() -> None:
    """Max on the findings that occupy a few slices, top2 on the ligaments, mean on the diffuse ones."""
    policy = dict(zip(TARGETS, WindowPooling.focal().per_target, strict=True))
    assert policy["Fracture"] == "max"
    assert policy["Contusion"] == "max"
    assert policy["Medial Meniscus"] == "max"
    assert policy["ACL"] == "top2"
    assert policy["MCL"] == "top2"
    assert policy["Synovitis"] == "mean"
    assert policy["Medial OA"] == "mean"


def test_pooled() -> None:
    """Each target is read by the reducer it was given, and the default is the mean this path had.

    THE MEAN ARM IS AN EQUALITY ORACLE, not a smoke test: every array this project has ever served was
    computed as a mean over windows, so a default that is anything else silently re-prices every member
    already on the roster.
    """
    # FOUR windows, chosen so the four reducers give four different answers: a three-window stack
    # whose centre is also its maximum cannot tell `centre` from `max`, and the first draft did not.
    levels = (0.1, 0.4, 0.6, 0.9)
    windows = torch.tensor([[[level] * len(TARGETS) for level in levels]])
    assert torch.allclose(WindowPooling().pooled(windows), windows.mean(dim=1))
    assert torch.allclose(WindowPooling.uniform("max").pooled(windows), torch.full((1, len(TARGETS)), 0.9))
    assert torch.allclose(WindowPooling.uniform("top2").pooled(windows), torch.full((1, len(TARGETS)), 0.75))
    assert torch.allclose(WindowPooling.uniform("centre").pooled(windows), torch.full((1, len(TARGETS)), 0.6))

    # A DIFFERENT COLUMN PER TARGET is the whole point, and a policy that reads one column with another
    # column's reducer is the defect a uniform arm cannot see.
    rising = torch.arange(len(TARGETS), dtype=torch.float32)
    mixed = torch.stack([rising, rising * 3.0, rising * 2.0]).unsqueeze(0)
    pooled = WindowPooling.focal().pooled(mixed)
    fracture, acl, synovitis = TARGETS.index("Fracture"), TARGETS.index("ACL"), TARGETS.index("Synovitis")
    assert pooled[0, fracture] == pytest.approx(rising[fracture].item() * 3.0)
    assert pooled[0, acl] == pytest.approx(rising[acl].item() * 2.5)
    assert pooled[0, synovitis] == pytest.approx(rising[synovitis].item() * 2.0)


def test_window_pooling_identity() -> None:
    """Whether a policy is the plain mean, which is what says an older array may stand in for it."""
    assert WindowPooling().identity
    assert WindowPooling.uniform("mean").identity
    assert not WindowPooling.focal().identity
    assert not WindowPooling.uniform("max").identity
