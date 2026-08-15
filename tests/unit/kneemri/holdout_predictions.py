"""Loading a member from a checkpoint and reading it over test-time windows, without a network.

The backbone is injected precisely so these can run: an encoder fetched inside `load` would make both
methods require a download, and a method that cannot be constructed in a test is a method whose contract
nothing checks.
"""

from pathlib import Path
from typing import override

import numpy as np
import torch

from kneemri.holdout_predictions import GROUP, HoldoutPredictions, Window
from kneemri.holdout_slab import HoldoutSlab
from kneemri.members import Member
from kneemri.model import CLS_MEAN_FOCAL, TARGETS, StudyDesign, StudyModel
from kneemri.prediction_cache import PredictionCache
from kneemri.study_cache import CachedCorpus
from kneemri.test_time import SpatialViews

DIM = 12
"""At most SIZE: the double reads one feature per image ROW, so a wider feature than the image has
rows cannot be filled."""
SLOTS = 6
SLICES = 6
SIZE = 14
SEED = 2026


class _FakeBackbone(torch.nn.Module):
    """A stand-in encoder whose tokens vary across the feature dimension.

    Varying per dimension is the load-bearing part. An earlier double returned the same scalar in every
    feature, which the head's opening LayerNorm removes exactly, so its output did not depend on its
    pixels and no test using it could detect anything.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(DIM, DIM) for _ in range(2)])

    @override
    def forward(self, pixel_values: torch.Tensor) -> object:
        rows = pixel_values.mean(dim=(1, 3))[:, :DIM]
        tokens = rows[:, None, :].expand(-1, 5, -1) * torch.linspace(0.5, 1.5, DIM)
        return type("Output", (), {"last_hidden_state": tokens})()


def _member() -> StudyModel:
    torch.manual_seed(SEED)
    return StudyModel(_FakeBackbone(), DIM, SLOTS)


def test_load(tmp_path: Path):
    """Weights round-trip, and the pooling rule comes from the checkpoint rather than a default."""
    saved = _member()
    checkpoint = tmp_path / "member.pt"
    torch.save({"model": saved.state_dict(), "config": {"slice_pooling": "channels"}}, checkpoint)

    torch.manual_seed(SEED)
    loaded = HoldoutPredictions(torch.device("cpu")).load(checkpoint, _FakeBackbone(), DIM, SLOTS)

    images = torch.randint(0, 256, (2, SLOTS, GROUP, SIZE, SIZE), dtype=torch.uint8)
    mask = torch.ones(2, SLOTS)
    with torch.no_grad():
        assert torch.allclose(loaded.forward(images, mask, SIZE), saved.eval().forward(images, mask, SIZE))


def test_load_honours_a_checkpoint_trained_with_per_slice_pooling(tmp_path: Path):
    """A member saved as `max` must not be rebuilt as `channels` — they are different functions."""
    torch.manual_seed(SEED)
    saved = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max"))
    checkpoint = tmp_path / "max.pt"
    torch.save({"model": saved.state_dict(), "config": {"slice_pooling": "max"}}, checkpoint)

    torch.manual_seed(SEED)
    loaded = HoldoutPredictions(torch.device("cpu")).load(checkpoint, _FakeBackbone(), DIM, SLOTS)
    assert loaded.slice_pooling == "max"


def test_load_rebuilds_a_focally_pooled_member(tmp_path: Path):
    """The tool that decides which members we submit has to be able to load the ones we can train.

    `pool` fixes the head's WIDTH, and this path read every switch back except that one — so a member
    fitted with `cls_mean_focal` came back two parts wide and died in `load_state_dict`, which is to
    say it could not be ranked at all. Dropping `pool` from the rebuild makes this fail.
    """
    torch.manual_seed(SEED)
    saved = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(pool=CLS_MEAN_FOCAL))
    checkpoint = tmp_path / "focal.pt"
    torch.save({"model": saved.state_dict(), "config": {"pool": CLS_MEAN_FOCAL}}, checkpoint)

    loaded = HoldoutPredictions(torch.device("cpu")).load(checkpoint, _FakeBackbone(), DIM, SLOTS)

    assert loaded.pool == CLS_MEAN_FOCAL
    images = torch.randint(0, 256, (2, SLOTS, GROUP, SIZE, SIZE), dtype=torch.uint8)
    mask = torch.ones(2, SLOTS)
    with torch.no_grad():
        assert torch.allclose(loaded.forward(images, mask, SIZE), saved.eval().forward(images, mask, SIZE))


def test_predict():
    """Probabilities for every study, in [0, 1], one row per study and one column per target."""
    predictions = HoldoutPredictions(torch.device("cpu"))
    generator = torch.Generator().manual_seed(SEED)
    images = torch.randint(0, 256, (5, SLOTS, SLICES, SIZE, SIZE), generator=generator, dtype=torch.uint8).numpy()
    mask = torch.ones(5, SLOTS).numpy()

    scored = predictions.predict(_member().eval(), images, mask, Window(starts=[0, 3], width=GROUP, size=SIZE))

    assert scored.shape == (5, len(TARGETS))
    assert scored.min() >= 0.0
    assert scored.max() <= 1.0


def test_predict_over_one_window_matches_a_single_forward():
    """With a single window there is nothing to average, so it must equal the model's own sigmoid."""
    model = _member().eval()
    generator = torch.Generator().manual_seed(SEED)
    images = torch.randint(0, 256, (3, SLOTS, SLICES, SIZE, SIZE), generator=generator, dtype=torch.uint8)
    mask = torch.ones(3, SLOTS)

    scored = HoldoutPredictions(torch.device("cpu")).predict(
        model, images.numpy(), mask.numpy(), Window(starts=[0], width=GROUP, size=SIZE)
    )

    with torch.no_grad():
        direct = torch.sigmoid(model.forward(images[:, :, 0:GROUP], mask, SIZE))
    assert torch.allclose(torch.from_numpy(scored).float(), direct, atol=1e-5)


def test_predict_averages_over_views():
    """Views multiply the passes averaged over, and the identity view alone must change nothing.

    The default is identity-only, so every number already in the ledger keeps its meaning. Real views
    matter because a whole-stack member reads ONE window: window averaging - worth a measured +0.0156
    to the windowed members - is structurally unavailable to it, leaving rotation as the only
    test-time average it can have.
    """
    model = _member().eval()
    generator = torch.Generator().manual_seed(SEED)
    images = torch.randint(0, 256, (3, SLOTS, SLICES, SIZE, SIZE), generator=generator, dtype=torch.uint8).numpy()
    mask = torch.ones(3, SLOTS).numpy()
    window = Window(starts=[0], width=GROUP, size=SIZE)

    plain = HoldoutPredictions(torch.device("cpu")).predict(model, images, mask, window)
    identity = HoldoutPredictions(torch.device("cpu"), SpatialViews(degrees=(0.0,), shifts=((0.0, 0.0),))).predict(
        model, images, mask, window
    )
    rotated = HoldoutPredictions(torch.device("cpu"), SpatialViews()).predict(model, images, mask, window)

    assert (plain == identity).all(), "the identity view must be exactly the un-augmented answer"
    assert rotated.shape == plain.shape
    assert not (rotated == plain).all(), "rotating the input must move the answer, or the average is a no-op"


def _reading(tmp_path: Path) -> Member:
    """A member whose checkpoint exists on disk, because the cache key stats it."""
    checkpoint = tmp_path / "m_x.pt"
    checkpoint.write_bytes(b"weights")
    return Member(
        identifier="x",
        file=checkpoint,
        fold=0,
        config={
            "img": 336,
            "slices": 12,
            "group": 3,
            "crop_mm": 130.0,
            "band": [0.2, 0.8],
            "slots": ["a", "b"],
            "variant": "small",
            "pool": "cls_mean",
        },
        fingerprint=np.zeros((2, 12), np.float32),
        holdout=0.8,
        annot=0.8,
    )


def test_cached(tmp_path: Path):
    """A cached reading comes back without materialising the pixels — which is the point of caching at all.

    The slab is left UNTOUCHED and that is asserted directly: everything expensive on this path sits
    downstream of `slab.arrays()` — the several-GB copy, the encoder download, the GPU — so a slab that
    was never materialised is proof that none of them ran. Without it the test would pass on any machine
    that happens to have a free card, which is the state this cache exists to stop depending on.
    """
    reading = _reading(tmp_path)
    cache = PredictionCache(tmp_path / "cached", ["s1", "s2"], SpatialViews(degrees=(0.0,), shifts=((0.0, 0.0),)))
    window = Window(starts=[0], width=3, size=336)
    stored = np.array([[0.25, 0.75], [0.5, 0.5]])
    cache.write(cache.key(reading.file, window, reading.precision), stored)
    slab = HoldoutSlab(CachedCorpus(np.zeros((2, 1, 1, 1, 1), np.uint8), np.ones((2, 1), np.float32)), np.array([0, 1]))

    scored = HoldoutPredictions(torch.device("cpu")).cached(reading, slab, cache, window)

    assert (scored == stored).all()
    assert slab.held is None, "a cache hit must not reach the pixels, and so must not reach the encoder"
