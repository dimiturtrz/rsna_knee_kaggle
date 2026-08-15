from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from torch import Tensor

from kneemri.model import TARGETS, StudyDesign, StudyModel
from kneemri.objective import Objective, RankedObjective
from kneemri.report_supervision import ReportSupervision
from kneemri.slot_head import REPORT_DIMS
from kneemri.study_cache import CachedCorpus
from kneemri.targets import Targets
from kneemri.trainer import GROUP, Adaptation, AugmentConfig, DeviceBudget, Trainer

SEED = 2026
DIM = 8
PATCHES = 16
SLOTS = 6


@dataclass
class _Output:
    last_hidden_state: Tensor


class _FakeBackbone(torch.nn.Module):
    """A stand-in encoder whose tokens vary PER DIMENSION with the input.

    An earlier version returned `base + mean(pixels)` — the same scalar added to every feature. The
    head begins with a LayerNorm, which removes exactly that, so the stand-in made the model's output
    independent of its pixels and no test could detect a change in pooling. A useful double has to
    perturb the feature vector's SHAPE, not just its level.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(DIM, DIM) for _ in range(12)])

    @override
    def forward(self, *, pixel_values: Tensor) -> _Output:
        n = pixel_values.shape[0]
        rows = pixel_values.mean(dim=(1, 3))[:, :DIM] if pixel_values.shape[2] >= DIM else None
        if rows is None:
            rows = pixel_values.mean(dim=(1, 3)).repeat(1, (DIM // pixel_values.shape[2]) + 1)[:, :DIM]
        base = torch.arange(PATCHES + 1, dtype=torch.float32).reshape(1, -1, 1) * 0.01
        return _Output(last_hidden_state=base + rows.reshape(n, 1, DIM))


def _trainer() -> Trainer:
    torch.manual_seed(SEED)
    return Trainer(StudyModel(_FakeBackbone(), DIM, SLOTS), torch.device("cpu"), Adaptation(unfreeze_last=6))


def test_augment():
    """Jitter moves the pixels but never mirrors them — a flip would undo laterality normalisation."""
    trainer = _trainer()
    generator = torch.Generator().manual_seed(SEED)
    images = torch.rand(4, 3, 16, 16)
    warped = trainer.augment(images, generator)
    assert warped.shape == images.shape
    assert not torch.allclose(warped, images), "augmentation must actually perturb"
    assert float(warped.max()) <= 1.0
    flipped = torch.flip(images, dims=[-1])
    assert not torch.allclose(warped, flipped, atol=0.2), "no horizontal flip may be introduced"


def test_batches():
    """Evaluation keeps every position; training drops the partial batch so the graph has ONE shape."""
    trainer = _trainer()
    generator = np.random.default_rng(SEED)
    positions = np.arange(10, dtype=np.int64)

    ordered = trainer.batches(positions, 4, generator, shuffle=False)
    assert [len(batch) for batch in ordered] == [4, 4, 2], "the holdout keeps its remainder"
    assert np.concatenate(ordered).tolist() == positions.tolist()

    shuffled = trainer.batches(positions, 4, generator, shuffle=True)
    assert [len(batch) for batch in shuffled] == [4, 4], "one shape in, one compiled graph out"
    assert len(set(np.concatenate(shuffled).tolist())) == 8, "the dropped two are dropped, not repeated"

    # A batch larger than the data must still yield ONE batch rather than none, or an epoch does
    # nothing and reports a mean over an empty list.
    assert len(trainer.batches(positions, 32, generator, shuffle=True)) == 1


def test_loss():
    """Weight zero silences a target entirely, which is how an unmentioned finding must behave."""
    trainer = _trainer()
    logits = torch.zeros(2, len(TARGETS))
    scores = torch.ones(2, len(TARGETS))
    weights = torch.zeros(2, len(TARGETS))
    assert float(trainer.loss(logits, scores, weights)) == 0.0
    weights[0, 0] = 1.0
    assert float(trainer.loss(logits, scores, weights)) > 0.0


def _corpus(studies: int = 6, slices: int = 6, size: int = 14) -> CachedCorpus:
    generator = np.random.default_rng(SEED)
    images = generator.integers(0, 256, (studies, SLOTS, slices, size, size), dtype=np.uint8)
    mask = np.ones((studies, SLOTS), dtype=np.float32)
    return CachedCorpus(images, mask)


def _targets(studies: int = 6) -> Targets:
    generator = np.random.default_rng(SEED)
    scores = generator.random((studies, len(TARGETS))).astype(np.float32)
    weights = np.ones((studies, len(TARGETS)), dtype=np.float32)
    return Targets(scores, weights, [f"s{index}" for index in range(studies)])


def test_fit_epoch():
    """One pass returns a finite mean loss and actually moves the head's weights."""
    trainer = _trainer()
    before = trainer.model.head.out.weight.detach().clone()
    loss = trainer.fit_epoch(_corpus(), _targets(), np.arange(6, dtype=np.int64), 3, np.random.default_rng(SEED))
    assert np.isfinite(loss)
    assert not torch.allclose(before, trainer.model.head.out.weight), "a fitted epoch must change the head"


def test_predict():
    """Prediction returns one probability per target per study, in [0, 1]."""
    trainer = _trainer()
    predictions = trainer.predict(_corpus(), np.arange(6, dtype=np.int64), 3, np.random.default_rng(SEED))
    assert predictions.shape == (6, len(TARGETS))
    assert predictions.min() >= 0.0
    assert predictions.max() <= 1.0


def test_augment_bag():
    """Bags stay uint8 and keep their shape, and every slot is perturbed rather than passed through.

    This exists because `augment` was written, unit-tested and then never called by `fit_epoch` — the
    training loop ran with no augmentation at all, which the dead-code gate could not see (the method
    WAS referenced: by its test) and the mirror gate could not see (it HAD a test). A test on the caller
    is what catches that class of bug.
    """
    trainer = _trainer()
    torch.manual_seed(SEED)
    bag = torch.randint(0, 256, (2, SLOTS, 3, 16, 16), dtype=torch.uint8)
    out = trainer.augment_bag(bag)
    assert out.shape == bag.shape
    assert out.dtype == torch.uint8
    assert not torch.equal(out, bag), "augmentation must actually perturb the bag"


def test_fit_epoch_augments():
    """`fit_epoch` must route its pixels through augmentation — the bug was that it did not."""
    trainer = _trainer()
    seen: list[torch.Tensor] = []
    original = trainer.augment_bag
    trainer.augment_bag = lambda images: seen.append(images) or original(images)  # type: ignore[method-assign]
    trainer.fit_epoch(_corpus(), _targets(), np.arange(6, dtype=np.int64), 3, np.random.default_rng(SEED))
    assert seen, "fit_epoch called no augmentation at all"
    assert seen[0].dtype == torch.uint8


def test_one_cycle():
    """The cycle starts BELOW the peak, warms up to it, and decays far below it by the end.

    Comparing against the pre-attach rate would prove nothing: `max_lr` IS the constant rate, so a
    warmed-up schedule passes through exactly the value it started from. The contract is the SHAPE.
    """
    trainer = _trainer()
    peak = [group["lr"] for group in trainer.optimiser.param_groups]
    total_steps = 40
    trainer.one_cycle(total_steps=total_steps)
    assert trainer.schedule is not None
    warmup_start = [group["lr"] for group in trainer.optimiser.param_groups]
    assert all(start < top for start, top in zip(warmup_start, peak, strict=True)), "must warm UP to the peak"

    for _ in range(total_steps):
        trainer.schedule.step()
    ended = [group["lr"] for group in trainer.optimiser.param_groups]
    assert all(end < top / 100 for end, top in zip(ended, peak, strict=True)), "must decay far below the peak"


def test_geometric_only():
    """The default must reproduce the pre-existing recipe exactly, or it silently changes members.

    Folds trained before the intensity block existed have to remain comparable with folds trained
    after it, so `geometric_only` is asserted to leave the intensity distribution alone: every pixel
    of an image scales by the SAME factor, which is what a gain is and a gamma/contrast/blur is not.
    """
    trainer = _trainer()
    trainer.augment_config = AugmentConfig.geometric_only()
    images = torch.rand(4, 3, 16, 16) * 0.8 + 0.1
    generator = torch.Generator().manual_seed(7)
    out = trainer._intensity(images, generator)
    ratio = out / images
    assert float(ratio.flatten(1).std(1).max()) < 1e-3, "a pure gain scales every pixel identically"


def test_intensity_perturbs_and_stays_in_range():
    """The full config must actually bite, and must not leave [0, 1] — the cache is uint8."""
    trainer = _trainer()
    trainer.augment_config = AugmentConfig()
    images = torch.rand(8, 3, 16, 16)
    out = trainer._intensity(images, torch.Generator().manual_seed(7))
    assert float((out - images).abs().gt(1e-4).float().mean()) > 0.5, "intensity augmentation must perturb"
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_loss_uses_the_objective_the_run_was_given():
    """The trainer must not carry its own copy of the rule — the objective decides, and it is held.

    Written after `normalise_by_count` moved onto the objective: the old test set that attribute on the
    TRAINER, where it now means nothing at all, and would have gone on passing while proving nothing.
    """
    # The targets must DIFFER between the two studies: identical readings separate no pair, so the
    # ranking term would be zero and this test would compare the default objective against itself.
    logits = torch.zeros(2, len(TARGETS))
    scores = torch.cat([torch.ones(1, len(TARGETS)), torch.zeros(1, len(TARGETS))])
    weights = torch.full((2, len(TARGETS)), 0.25)

    torch.manual_seed(SEED)
    ranked = Trainer(StudyModel(_FakeBackbone(), DIM, SLOTS), torch.device("cpu"), objective=RankedObjective(1.0))
    assert float(ranked.loss(logits, scores, weights)) != float(_trainer().loss(logits, scores, weights))
    assert float(_trainer().loss(logits, scores, weights)) == float(Objective()(logits, scores, weights)), (
        "the default trainer must descend plain weighted cross-entropy"
    )


def test_save_partial(tmp_path: Path):
    """The best weights hit disk the moment they improve, and a partial write cannot survive.

    The payload is now assembled by the caller and is the SAME one the final save writes — a partial
    that carried only weights preserved a killed run's work but not its identity, and MemberPackage
    refuses a member with no fingerprint.
    """
    trainer = _trainer()
    destination = tmp_path / "member_fold0.pt"
    state = {key: value.detach().cpu() for key, value in trainer.model.head.state_dict().items()}
    trainer.save_partial(destination, {"model": state, "holdout": 0.81, "epoch": 7, "config": {"slice_pooling": "max"}})

    partial = destination.with_suffix(".partial")
    assert partial.exists()
    assert not partial.with_suffix(".tmp").exists(), "the temp file must be renamed, not left behind"
    reloaded = torch.load(partial, map_location="cpu", weights_only=True)
    assert reloaded["holdout"] == 0.81
    assert reloaded["epoch"] == 7
    assert reloaded["model"].keys() == state.keys()
    assert reloaded["config"]["slice_pooling"] == "max", "the reading travels with the weights"


def test_exponential_average():
    """With an EMA attached, the saved weights lag the live ones instead of being them."""
    trainer = _trainer()
    trainer.exponential_average()
    trainer.fit_epoch(_corpus(), _targets(), np.arange(6, dtype=np.int64), 3, np.random.default_rng(SEED))

    live = trainer.model.head.out.weight.detach().clone()
    averaged = trainer.average.state_dict(trainer.model)["head.out.weight"]
    assert not torch.allclose(live, averaged), "an EMA that equals the live weights is not averaging"


def test_evaluated():
    """The block scores under the averaged weights and puts the live ones back afterwards.

    Leaving the average in place would resume training from smoothed weights, which is a different
    optimisation entirely - and one that compounds every epoch.
    """
    trainer = _trainer()
    trainer.exponential_average()
    trainer.fit_epoch(_corpus(), _targets(), np.arange(6, dtype=np.int64), 3, np.random.default_rng(SEED))

    live = trainer.model.head.out.weight.detach().clone()
    with trainer.evaluated():
        inside = trainer.model.head.out.weight.detach().clone()
    assert not torch.allclose(inside, live), "scoring must happen under the averaged weights"
    assert torch.equal(trainer.model.head.out.weight.detach(), live), "the live weights must be restored"


def test_evaluated_without_an_average_is_a_no_op():
    """The default policy leaves the weights alone, so a run without an EMA pays nothing."""
    trainer = _trainer()
    before = trainer.model.head.out.weight.detach().clone()
    with trainer.evaluated():
        assert torch.equal(trainer.model.head.out.weight.detach(), before)
    assert torch.equal(trainer.model.head.out.weight.detach(), before)


def test_report_supervision():
    """Attaching report supervision changes the loss a step descends, and the default does not.

    The term is an auxiliary on the study representation, so it has to reach the loss without the
    training loop asking whether it is switched on.
    """
    torch.manual_seed(SEED)
    vectors = np.random.default_rng(SEED).normal(size=(6, REPORT_DIMS)).astype(np.float32)

    plain = _trainer()
    plain_loss = plain.fit_epoch(_corpus(), _targets(), np.arange(6, dtype=np.int64), 3, np.random.default_rng(SEED))

    supervised = _trainer()
    supervised.report_supervision(ReportSupervision(vectors, weight=1.0))
    supervised_loss = supervised.fit_epoch(
        _corpus(), _targets(), np.arange(6, dtype=np.int64), 3, np.random.default_rng(SEED)
    )

    assert supervised_loss != plain_loss, "the auxiliary term must reach the loss"
    assert np.isfinite(supervised_loss)


def test_whole_stack():
    """Reading the whole stack changes what a step sees, and is refused under channel stacking.

    Channel stacking has exactly three channels to stack into, so a wider window is not a preference
    there - it is a shape error waiting to happen. Per-slice pooling encodes each slice alone, which is
    what makes any window width meaningful.
    """
    torch.manual_seed(SEED)
    stacked = Trainer(StudyModel(_FakeBackbone(), DIM, SLOTS), torch.device("cpu"))
    with pytest.raises(ValueError, match="per-slice pooling"):
        stacked.whole_stack(6)

    torch.manual_seed(SEED)
    per_slice = Trainer(StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max")), torch.device("cpu"))
    assert per_slice.window == GROUP
    per_slice.whole_stack(6)
    assert per_slice.window == 6

    loss = per_slice.fit_epoch(_corpus(), _targets(), np.arange(6, dtype=np.int64), 3, np.random.default_rng(SEED))
    assert np.isfinite(loss), "a whole-stack step must still train"


def test_bfloat16():
    """bf16 carries fp32's exponent range, so a GradScaler is not merely unneeded — it is a second
    thing that can go wrong for no gain, and it is disabled rather than left enabled and idle.

    fp16 stays the DEFAULT because every number in the ledger was measured under it, and a precision
    change is a numerics change: a run under bf16 is not strictly matched to an fp16 control.
    """
    torch.manual_seed(SEED)
    default = Trainer(StudyModel(_FakeBackbone(), DIM, SLOTS), torch.device("cpu"))
    assert default.autocast_dtype is torch.float16

    torch.manual_seed(SEED)
    wide = Trainer(StudyModel(_FakeBackbone(), DIM, SLOTS), torch.device("cpu"))
    wide.bfloat16()
    assert wide.autocast_dtype is torch.bfloat16
    assert not wide.scaler.is_enabled()

    loss = wide.fit_epoch(_corpus(), _targets(), np.arange(6, dtype=np.int64), 3, np.random.default_rng(SEED))
    assert np.isfinite(loss), "a bf16 step must still train"


def test_resample_to():
    """Resampling changes the size the model is READ at, and leaves the cache's own size alone.

    The contract that matters is the default: an untouched trainer must pass the corpus's resolution
    through unchanged, because every member fitted before resampling existed means exactly that. If
    `read_size` defaulted to anything but None, those members would be rebuilt at a size they were
    never fitted under — the failure `slice_pooling` already caused once.
    """
    torch.manual_seed(SEED)
    trainer = Trainer(StudyModel(_FakeBackbone(), DIM, SLOTS), torch.device("cpu"))
    assert trainer.read_size is None, "an untouched trainer reads the cache at its own resolution"

    trainer.resample_to(392)
    assert trainer.read_size == 392

    loss = trainer.fit_epoch(_corpus(), _targets(), np.arange(6, dtype=np.int64), 3, np.random.default_rng(SEED))
    assert np.isfinite(loss), "a resampled step must still train"


def test_select():
    """An uncapped budget still yields a device; the cap is applied only where there is a card.

    The unit suite runs without a GPU, so this pins the contract that matters here — `select` is the
    one place a run's device is decided, and it must work with no CUDA present rather than being a
    CUDA-only path that silently never runs.
    """
    device = DeviceBudget().select()

    assert device.type in {"cuda", "cpu"}
    assert DeviceBudget(vram_gb=8.0).select().type == device.type, "a budget must not change WHICH device"


def test_plan():
    """Drawing every window start up front yields exactly what drawing one per step yielded.

    The prefetch reads ahead of the loop, so the schedule had to leave it — and this is the check that
    the move cost nothing: same seed, same numbers, in the same order. If a draw is ever added between
    the starts, this fails rather than silently retraining every arm on a different sequence.
    """
    corpus = CachedCorpus(np.zeros((8, 2, 6, 4, 4), dtype=np.uint8), np.ones((8, 2), dtype=np.float32))
    trainer = _trainer()
    batches = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]

    planned = trainer.plan(corpus, batches, np.random.default_rng(SEED))
    per_step = np.random.default_rng(SEED)
    for (batch, start), expected in zip(planned, batches, strict=True):
        assert np.array_equal(batch, expected)
        assert start == int(per_step.integers(0, corpus.slices - trainer.window + 1))
