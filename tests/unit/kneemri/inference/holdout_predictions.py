"""Loading a member from a checkpoint and reading it over test-time windows, without a network.

The backbone is injected precisely so these can run: an encoder fetched inside `load` would make both
methods require a download, and a method that cannot be constructed in a test is a method whose contract
nothing checks.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch

from core.data.corpus.slot_reader import GROUP
from core.findings import TARGETS
from kneemri.inference.holdout_predictions import HoldoutPredictions, ServedModel, Window
from kneemri.inference.holdout_slab import HoldoutSlab
from kneemri.inference.members import Member
from kneemri.inference.prediction_cache import PredictionCache
from kneemri.inference.test_time import SpatialViews
from kneemri.models.model import StudyModel
from kneemri.models.study_design import StudyDesign
from kneemri.models.token_pooling import CLS_MEAN_FOCAL

DIM = 12
"""At most SIZE: the double reads one feature per image ROW, so a wider feature than the image has
rows cannot be filled."""
SLOTS = 6
SLICES = 6
SIZE = 14
SEED = 2026

PLAIN = SpatialViews(degrees=(0.0,), shifts=((0.0, 0.0),))
"""One unwarped pass — the reading every ledger number was taken under."""


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


def test_fits():
    """A window that runs past the served stack is refused; one that fits passes silently.

    Both faces of knee-kcvma are exercised, because only one of them ever announced itself. A window
    starting BEYOND the depth yields an empty view and raised deep inside the encoder as a reshape of
    zero elements, naming neither the window nor the slab. A window that merely OVERHANGS the end
    raised nothing at all and scored six members on eight slices of twenty. The second is why this is a
    check rather than a comment.
    """
    HoldoutPredictions.fits(Window(starts=[0], width=SLICES, size=SIZE), SLICES)
    HoldoutPredictions.fits(Window(starts=[0, 2], width=GROUP, size=SIZE), SLICES)

    with pytest.raises(ValueError, match="does not fit a stack"):
        HoldoutPredictions.fits(Window(starts=[SLICES + 4], width=SLICES, size=SIZE), SLICES)

    with pytest.raises(ValueError, match="does not fit a stack"):
        HoldoutPredictions.fits(Window(starts=[SLICES - 2], width=SLICES, size=SIZE), SLICES)

    with pytest.raises(ValueError, match="does not fit a stack"):
        HoldoutPredictions.fits(Window(starts=[0], width=0, size=SIZE), SLICES)


def test_autocast():
    """The member's own precision on the card, and no autocast at all off it."""
    served = ServedModel(_member(), Window(starts=[0], width=GROUP, size=SIZE), torch.bfloat16)
    assert served.autocast(torch.device("cuda")).fast_dtype == torch.bfloat16
    with served.autocast(torch.device("cpu")):
        assert not torch.is_autocast_enabled("cuda")
        assert not torch.is_autocast_enabled("cpu")


def test_crop():
    """Exactly `window.width` slices from the start asked, never the window's own first start."""
    stack = torch.arange(SLICES, dtype=torch.uint8).view(1, 1, SLICES, 1, 1).expand(2, SLOTS, SLICES, 3, 3)
    served = ServedModel(_member(), Window(starts=[0], width=GROUP, size=SIZE))
    assert served.crop(stack, 2)[0, 0, :, 0, 0].tolist() == [2, 3, 4]


def test_logits():
    """One forward at one start, equal to calling the model on the cropped stack at the member's size."""
    model = _member().eval()
    generator = torch.Generator().manual_seed(SEED)
    images = torch.randint(0, 256, (2, SLOTS, SLICES, SIZE, SIZE), generator=generator, dtype=torch.uint8)
    mask = torch.ones(2, SLOTS)
    served = ServedModel(model, Window(starts=[0, 3], width=GROUP, size=SIZE))
    with torch.no_grad():
        assert torch.equal(served.logits(images, mask, 3), model(images[:, :, 3 : 3 + GROUP], mask, SIZE))


def test_predict():
    """Probabilities for every study, in [0, 1], one row per study and one column per target."""
    predictions = HoldoutPredictions(torch.device("cpu"))
    generator = torch.Generator().manual_seed(SEED)
    images = torch.randint(0, 256, (5, SLOTS, SLICES, SIZE, SIZE), generator=generator, dtype=torch.uint8).numpy()
    mask = torch.ones(5, SLOTS).numpy()

    scored = predictions.predict(
        _member().eval(), HoldoutSlab.of(images, mask), Window(starts=[0, 3], width=GROUP, size=SIZE)
    )

    assert scored.shape == (5, len(TARGETS))
    assert scored.min() >= 0.0
    assert scored.max() <= 1.0


class _SpyAdaptation:
    """Records what the reading loop hands an adaptation, and adapts nothing."""

    def __init__(self, episode: int) -> None:
        self._episode = episode
        self.studies: list[int] = []
        self.graded: list[torch.Tensor] = []

    @property
    def episode(self) -> int:
        return self._episode

    @contextmanager
    def on(self, member: object, logits: Callable[[], torch.Tensor]) -> Iterator[None]:
        del member
        graded = logits()
        self.studies.append(graded.shape[0])
        self.graded.append(graded)
        yield


def test_predict_takes_its_batch_size_from_the_adaptations_episode():
    """An episodic recipe must not adapt eight studies at once, and BATCH is this loop's, not its.

    Eight studies sharing one gradient step moves the affines that read study N+1 by study N's
    findings — the same neighbour leak the per-study top-k already had to fix, one level up. The
    default declares no episode and must leave the batching exactly as every board figure took it.
    """
    generator = torch.Generator().manual_seed(SEED)
    images = torch.randint(0, 256, (5, SLOTS, SLICES, SIZE, SIZE), generator=generator, dtype=torch.uint8).numpy()
    mask = torch.ones(5, SLOTS).numpy()
    window = Window(starts=[0], width=GROUP, size=SIZE)

    episodic, unbounded = _SpyAdaptation(1), _SpyAdaptation(0)
    for spy in (episodic, unbounded):
        HoldoutPredictions(torch.device("cpu"), adaptation=spy).predict(
            _member().eval(), HoldoutSlab.of(images, mask), window
        )

    assert episodic.studies == [1, 1, 1, 1, 1]
    assert unbounded.studies == [5]


def test_predict_hands_the_adaptation_a_graded_read_of_the_right_shape():
    """The closure must carry a gradient and one row per study, or the adapt pass is a silent no-op.

    Every serving loop here is `no_grad`-wrapped and this one used to wrap the whole study walk. If
    that wrapper still enclosed the adapt pass, `logits()` would come back detached, the backward
    would raise or do nothing, and the reading would carry an adapted name with unadapted values.
    """
    generator = torch.Generator().manual_seed(SEED)
    images = torch.randint(0, 256, (3, SLOTS, SLICES, SIZE, SIZE), generator=generator, dtype=torch.uint8).numpy()
    spy = _SpyAdaptation(0)

    HoldoutPredictions(torch.device("cpu"), adaptation=spy).predict(
        _member().eval(),
        HoldoutSlab.of(images, torch.ones(3, SLOTS).numpy()),
        Window(starts=[0], width=GROUP, size=SIZE),
    )

    assert spy.graded[0].shape == (3, len(TARGETS))
    assert spy.graded[0].requires_grad


def test_predict_over_one_window_matches_a_single_forward():
    """With a single window there is nothing to average, so it must equal the model's own sigmoid."""
    model = _member().eval()
    generator = torch.Generator().manual_seed(SEED)
    images = torch.randint(0, 256, (3, SLOTS, SLICES, SIZE, SIZE), generator=generator, dtype=torch.uint8)
    mask = torch.ones(3, SLOTS)

    scored = HoldoutPredictions(torch.device("cpu")).predict(
        model, HoldoutSlab.of(images.numpy(), mask.numpy()), Window(starts=[0], width=GROUP, size=SIZE)
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

    plain = HoldoutPredictions(torch.device("cpu")).predict(model, HoldoutSlab.of(images, mask), window)
    identity = HoldoutPredictions(torch.device("cpu"), PLAIN).predict(model, HoldoutSlab.of(images, mask), window)
    rotated = HoldoutPredictions(torch.device("cpu"), SpatialViews()).predict(
        model, HoldoutSlab.of(images, mask), window
    )

    assert (plain == identity).all(), "the identity view must be exactly the un-augmented answer"
    assert rotated.shape == plain.shape
    assert not (rotated == plain).all(), "rotating the input must move the answer, or the average is a no-op"


def test_cached(tmp_path: Path, reading: Callable[[Path], Member], unreadable_slab: Callable[[int], HoldoutSlab]):
    """A cached reading comes back without materialising the pixels — which is the point of caching at all.

    The slab is one that CANNOT be read — its holdout names rows the corpus does not have — so a hit
    that touched the pixels raises rather than passing quietly. Everything expensive on this path sits
    downstream of that read: the encoder download and the GPU. Without the trap the test would pass on
    any machine that happens to have a free card, which is the state this cache exists to stop
    depending on.
    """
    member = reading(tmp_path)
    cache = PredictionCache(tmp_path / "cached", ["s1", "s2"], PLAIN)
    window = Window(starts=[0], width=3, size=336)
    stored = np.array([[0.25, 0.75], [0.5, 0.5]])
    cache.write(cache.key(member.file, window, member.precision), stored)
    unreadable = unreadable_slab(1)

    scored = HoldoutPredictions(torch.device("cpu")).cached(member, unreadable, cache, window)

    assert (scored == stored).all()
    with pytest.raises(IndexError):
        unreadable.batch(0, 2)


def test_cached_prefers_the_record_over_the_card(
    tmp_path: Path, recorded: Callable[..., Member], unreadable_slab: Callable[[int], HoldoutSlab]
):
    """With no cache entry and no GPU, a recorded member still answers — and the answer is then cached.

    The slab holds no pixels at all, so reaching the encoder raises. That is the assertion: before this,
    every uncached member cost a full inference pass, measured at over fifteen minutes on a ConvNeXtV2
    base, which is what made a ninety-five member sweep unfinishable.
    """
    member = recorded(tmp_path, ["s1", "s2"])
    cache = PredictionCache(tmp_path / "prefer", ["s2", "s1"], PLAIN)
    window = member.window(SLICES, overlap=False)

    scored = HoldoutPredictions(torch.device("cpu")).cached(member, unreadable_slab(SLICES), cache, window)

    assert scored == pytest.approx(np.array([[0.2, 0.3], [0.0, 0.1]]))
    assert cache.read(cache.key(member.file, window, member.precision)) == pytest.approx(scored)
