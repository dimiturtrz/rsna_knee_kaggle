import logging
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from torch import Tensor

from kneemri.corpus.slot_reader import GROUP
from kneemri.corpus.study_cache import CachedCorpus
from kneemri.device import VisibleCards
from kneemri.findings import TARGETS
from kneemri.labels.targets import Targets
from kneemri.models.backbone import ENCODERS, SMALL, V3_BASE, V3_SMALL
from kneemri.models.model import StudyDesign, StudyModel
from kneemri.models.slot_head import REPORT_DIMS
from kneemri.training.augmentation import INTENSITY_AUGMENT
from kneemri.training.objective import (
    COUNT_NORMALISED,
    PAIR_POOL,
    PER_LABEL,
    RANK_TERM,
    Objective,
    Predicted,
    RankedObjective,
)
from kneemri.training.report_supervision import ReportSupervision
from kneemri.training.trainer import (
    ACCUMULATION,
    AUTOCAST,
    AVERAGING,
    CONTENTION_POLL_SECONDS,
    CONTENTION_WAIT_SECONDS,
    DEFAULT_SHARE,
    FITTED_BATCH,
    READ_SIZE,
    REPORTS,
    RESIDENT_GB,
    SCHEDULED,
    SMALLEST_RUN_GB,
    WINDOW,
    Adaptation,
    DeviceBudget,
    Trainer,
)

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
    objective = _trainer().objective
    logits = torch.zeros(2, len(TARGETS))
    scores = torch.ones(2, len(TARGETS))
    weights = torch.zeros(2, len(TARGETS))
    assert float(objective(Predicted(logits, scores, weights))) == 0.0
    weights[0, 0] = 1.0
    assert float(objective(Predicted(logits, scores, weights))) > 0.0


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
    assert isinstance(loss, float), "the epoch mean is a host number — the running total is what stays on device"
    assert not torch.allclose(before, trainer.model.head.out.weight), "a fitted epoch must change the head"


def test_accumulate():
    """Attaching the policy is visible in the trainer's own account of itself."""
    trainer = _trainer()
    assert trainer.policies()[ACCUMULATION] == 1, "the default loop steps on every micro-batch"
    trainer.accumulate(4)
    assert trainer.policies()[ACCUMULATION] == 4


def test_fit_epoch_accumulation_matches_the_larger_batch():
    """THE ORACLE FOR THE WHOLE FEATURE: two micro-batches of three must land where one batch of six does.

    Accumulation exists to buy an effective batch the card cannot hold, and the only thing that claim
    means is that the WEIGHTS after the step are the big batch's weights. Anything less — a missing
    division, a zero_grad inside the group, a schedule stepped per micro-batch — moves them, and each
    of those failures still trains, still logs, and still reports a number under the arm's name.

    Two things are held fixed so the comparison is about the gradient and nothing else: the corpus is
    exactly one window deep, so the per-batch window draw has one outcome and both arms read the same
    pixels; and augmentation is bypassed, because it draws from the trainer's generator per batch and
    two draws are not one.
    """
    weights = []
    for size, every in ((6, 1), (3, 2)):
        trainer = _trainer()
        trainer.accumulate(every)
        trainer.augment_bag = lambda images: images  # type: ignore[method-assign]
        corpus = _corpus(slices=GROUP)
        trainer.fit_epoch(corpus, _targets(), np.arange(6, dtype=np.int64), size, np.random.default_rng(SEED))
        weights.append(trainer.model.head.out.weight.detach().clone())

    assert torch.allclose(weights[0], weights[1], atol=1e-6), (
        "accumulated micro-batches must step the optimiser exactly where the whole batch would"
    )


def test_predict():
    """Prediction returns one probability per target per study, in [0, 1] — remainder included.

    The last batch is padded up to the batch size and trimmed after the forward, so a count that is NOT
    a multiple of the size is the case that proves the trim: a holdout short of its remainder scores a
    different set of studies than the ledger quotes.
    """
    trainer = _trainer()
    predictions = trainer.predict(_corpus(), np.arange(6, dtype=np.int64), 3, np.random.default_rng(SEED))
    assert predictions.shape == (6, len(TARGETS))
    assert predictions.min() >= 0.0
    assert predictions.max() <= 1.0

    remainder = trainer.predict(_corpus(), np.arange(5, dtype=np.int64), 3, np.random.default_rng(SEED))
    assert remainder.shape == (5, len(TARGETS))
    assert np.allclose(remainder[:3], predictions[:3]), "padding the tail must not disturb the full batches"


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


def test_policies():
    """The trainer's own account of which optional policies are attached, before and after attaching.

    Each absence is a working no-op object rather than a crash — an EMA that was never asked for is a
    `LiveWeights`, and the run trains fine and reports a number for a recipe nobody asked for. The
    report flag reads the SUBCLASS, not a boolean, so it cannot be true while the no-op is in place.

    The loss switches come from the objective rather than from here, so this asserts only that they are
    PASSED THROUGH — `tests/unit/kneemri/training/objective.py` owns what each objective reports.
    """
    trainer = Trainer(
        StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max")),
        torch.device("cpu"),
        objective=RankedObjective(0.3),
    )
    constructed = {
        RANK_TERM: 0.3,
        PAIR_POOL: 0,
        PER_LABEL: False,
        COUNT_NORMALISED: False,
        INTENSITY_AUGMENT: False,
    }

    assert trainer.policies() == {
        **constructed,
        AUTOCAST: torch.float16,
        WINDOW: GROUP,
        READ_SIZE: None,
        AVERAGING: False,
        SCHEDULED: False,
        REPORTS: False,
        ACCUMULATION: 1,
    }

    trainer.bfloat16()
    trainer.whole_stack(6)
    trainer.resample_to(392)
    trainer.exponential_average()
    trainer.one_cycle(total_steps=40)

    assert trainer.policies() == {
        **constructed,
        AUTOCAST: torch.bfloat16,
        WINDOW: 6,
        READ_SIZE: 392,
        AVERAGING: True,
        SCHEDULED: True,
        REPORTS: False,
        ACCUMULATION: 1,
    }


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


def test_fit_epoch_steps_the_schedule():
    """`fit_epoch` must advance the cycle once per optimiser step, or `--one-cycle` is a flag that does nothing.

    The shipping recipe carries `--one-cycle`, and the only test of it drove the schedule BY HAND — the same
    shape that let `augment` be written, tested and never called. Per STEP, not per epoch: `one_cycle` is
    built with `total_steps`, so an epoch-rate schedule would sit at the warm-up rate for the whole run.
    """
    trainer = _trainer()
    positions = np.arange(6, dtype=np.int64)
    size = 3
    trainer.one_cycle(total_steps=len(positions) // size)
    assert trainer.schedule is not None
    started = [group["lr"] for group in trainer.optimiser.param_groups]

    trainer.fit_epoch(_corpus(), _targets(), positions, size, np.random.default_rng(SEED))

    assert trainer.schedule.last_epoch == len(positions) // size, "one schedule step per batch, not per epoch"
    ended = [group["lr"] for group in trainer.optimiser.param_groups]
    assert ended != started, "a schedule the loop never steps leaves the rate where it started"


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
    judged = Predicted(logits, scores, weights)
    assert float(ranked.objective(judged)) != float(_trainer().objective(judged))
    assert float(_trainer().objective(judged)) == float(Objective()(judged)), (
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


def test_floor():
    """The refusal threshold is the cheapest run on record, or the run's own budget when that is lower.

    IT IS DELIBERATELY NOT THE CAP. The cap defaults to 0.8 of the card, and refusing on that would
    refuse whenever the co-tenant holds a fifth of it — common, and harmless to a 5.5 GB smoke.
    """
    assert DeviceBudget().floor() == SMALLEST_RUN_GB, "silence is refused only below our cheapest run"
    assert DeviceBudget(vram_gb=26.0).floor() == SMALLEST_RUN_GB, "a big budget does not raise the floor"
    assert DeviceBudget(vram_gb=3.0).floor() == 3.0, "a run that claims to fit in less is taken at its word"


def test_select_refuses_a_card_that_stayed_held(monkeypatch: pytest.MonkeyPatch):
    """Past the wait's bound, a card short of the cheapest run is refused rather than started into.

    THE FAILURE THIS CLOSES cost both projects on 2026-08-17: starting into a held card does not raise
    on Windows, it pages over PCIe and crawls, and the pair of us then drew enough to reset the driver.
    A refusal is loud, immediate and re-runnable; the crawl was none of those.
    """
    monkeypatch.setattr(VisibleCards, "device", lambda _: torch.device("cuda", 0))
    monkeypatch.setattr(DeviceBudget, "await_headroom", lambda _: (1.0, 32.0))

    with pytest.raises(RuntimeError, match="Refusing rather than paging"):
        DeviceBudget().select()


def test_headroom():
    """Free VRAM is readable without a card, and free never exceeds the total.

    The suite runs on CPU, where the honest answer is (0, 0) — the point is that asking costs nothing
    and returns a pair rather than raising, since `select` reads it before every run on every machine.
    """
    free, whole = DeviceBudget().headroom()

    assert free <= whole, "free VRAM cannot exceed the card"
    assert (free, whole) == (0.0, 0.0) or whole > 0, "a visible card reports a positive total"


def test_await_headroom(monkeypatch: pytest.MonkeyPatch):
    """A held card is waited out, and a card held by something stuck is waited out only so far.

    THE CO-TENANT IS PERMANENT: a second project of the same owner trains on this same card, so
    contention is a condition to schedule around rather than an incident to report. Waiting costs a
    queued arm minutes; starting anyway cost, on 2026-08-17, three driver resets that killed BOTH
    projects' processes.

    The sleep is injected because this must not spend the wall clock it is describing.
    """
    readings = iter([(4.0, 32.0), (6.0, 32.0), (30.0, 32.0)])
    monkeypatch.setattr(DeviceBudget, "headroom", lambda _: next(readings))
    slept: list[float] = []

    free, whole = DeviceBudget(vram_gb=22.0).await_headroom(slept.append)

    assert (free, whole) == (30.0, 32.0), "it returns the headroom the run actually starts on"
    assert slept == [CONTENTION_POLL_SECONDS] * 2, "and polls only while the card is short"

    held = iter([(1.0, 32.0)] * 1000)
    monkeypatch.setattr(DeviceBudget, "headroom", lambda _: next(held))
    stalled: list[float] = []

    assert DeviceBudget(vram_gb=22.0).await_headroom(stalled.append) == (1.0, 32.0), "it gives up and says so"
    assert sum(stalled) == CONTENTION_WAIT_SECONDS, "a co-tenant that hung cannot hold the queue all night"

    idle: list[float] = []
    monkeypatch.setattr(DeviceBudget, "headroom", lambda _: (31.0, 32.0))

    assert DeviceBudget().await_headroom(idle.append) == (31.0, 32.0)
    assert not idle, "a run that named nothing has no need to wait ON, and the whole card is never free"


def test_headroom_without_a_card(monkeypatch: pytest.MonkeyPatch):
    """No visible card answers (0, 0) rather than asking the driver, which would raise.

    THE MACHINE THAT RUNS THIS SUITE HAS A CARD, so the branch that keeps a CPU box working is the one
    nothing here exercises — and `select` reads it before every run, on every machine.
    """
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    assert DeviceBudget().headroom() == (0.0, 0.0), "asking costs nothing and returns a pair"


def test_select_warns_when_it_starts_into_a_busy_card(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Past the wait's bound, a card that is short but not FATAL is started into — and says so.

    The symptom is not a failure and cannot be read off the throughput afterwards: Windows pages over
    PCIe and the run looks like a slow recipe. So the contention is named before the first batch,
    which is the whole difference between this and the crawl that cost both projects on 2026-08-17.
    """
    monkeypatch.setattr(VisibleCards, "device", lambda _: torch.device("cuda", 0))
    monkeypatch.setattr(DeviceBudget, "await_headroom", lambda _: (10.0, 32.0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    fractions: list[float] = []
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, _: fractions.append(fraction))

    with caplog.at_level(logging.WARNING):
        device = DeviceBudget().select()

    assert device.type == "cuda", "short of the cap is not short of the floor, so the run starts"
    assert "another process holds the rest" in caplog.text, "named before the first batch, not inferred after it"
    assert fractions == [pytest.approx(0.8)], "and the cap it starts under is still applied"


def test_capped():
    """A run that named a budget gets it; one that named nothing still gets a cap.

    THE SECOND HALF IS THE POINT. Uncapped was the default until now, which made this class's own
    worst failure — Windows paging VRAM over PCIe, so an over-committed run crawls at a reported 100%
    utilisation instead of raising — the behaviour a run got by SAYING NOTHING. Every accepted arm
    names its budget, so the default was never what a measured run did, only what a smoke or a
    one-off got.
    """
    assert DeviceBudget(vram_gb=26.0).capped(31.8) == 26.0, "an explicit budget is used as asked"
    assert DeviceBudget().capped(31.8) == pytest.approx(DEFAULT_SHARE * 31.8), "and silence still caps"
    assert DeviceBudget().capped(31.8) < 31.8, "the driver context and the desktop live outside the allocator"
    twelve = DeviceBudget(slices=12).capped(31.8)
    assert 16.24 < twelve < DEFAULT_SHARE * 31.8, "a stated recipe caps at its own need, under the share"
    assert DeviceBudget(vram_gb=28.0, slices=12).capped(31.8) == 28.0, "a named number still overrides it"
    # THE SHARE MUST NOT CLAMP A DERIVED NEED. 20 slices reserve 26.92 GB measured and the share is
    # 25.44, so the first version of `capped` — a `min` against the share — would have capped every
    # 20-slice arm below its own measured demand. Those arms RAN, at a hand-typed 28. A cap under a
    # measured need does not prevent the over-commit, it schedules it.
    assert DeviceBudget(slices=20).capped(31.8) > DEFAULT_SHARE * 31.8, "a measured need outranks the blind default"
    assert DeviceBudget(slices=20).capped(31.8) == pytest.approx(26.92 + 1.335, abs=0.02)


def test_needed():
    """The fitted line, checked at the point it was NOT fitted on.

    12 and 20 slices (16.24 and 26.92 GB reserved) give the coefficients; 16 was held out and measured
    21.57. The margin is one slice, so `needed` sits a slice above each measured peak rather than on it.
    """
    held_out_measured = 21.57
    one_slice = 1.335
    assert DeviceBudget(slices=16).needed() == pytest.approx(held_out_measured + one_slice, abs=0.02)
    assert DeviceBudget(slices=20).needed() > DeviceBudget(slices=12).needed(), "more slices, more memory"
    assert DeviceBudget().needed() == 0.0, "an unstated recipe predicts nothing rather than guessing"

    at_336 = DeviceBudget(slices=12).needed()
    assert DeviceBudget(slices=12, size=336).needed() == at_336, "the fit's own size is the default"
    assert DeviceBudget(slices=12, size=420).needed() == pytest.approx(at_336 * (420 / 336) ** 2, abs=0.3), (
        "a slice is a patch grid, so it costs the square of the side"
    )
    assert DeviceBudget(slices=12, size=420).needed() > 23.42, (
        "the cap must clear the rung MEASURED at 420 px, or it forbids a recipe that fits"
    )


def test_needed_reads_the_encoder():
    """The cap is a function of the ENCODER too, and every arm behind the fit must be untouched by it.

    `v3base_band` died in seconds on 2026-08-20 with `Tried to allocate 848.00 MiB ... 12.68 GiB
    allowed` while 17.51 GiB of the card was free: a `small`-fitted cap handed to a 768-wide encoder.
    The first assertion is the one that protects the ledger — the whole DINOv2 corpus is patch 14 and
    384 wide, so the new terms must reduce to exactly the old number for every arm already recorded.
    """
    for size in (224, 336, 392, 420):
        v2 = ENCODERS[SMALL]
        assert DeviceBudget(slices=12, size=size, patch=v2.patch, dim=v2.dim).needed() == pytest.approx(
            DeviceBudget(slices=12, size=size).needed()
        ), "naming small's own width and patch cannot move a number the arms were fitted with"

    small, base = ENCODERS[V3_SMALL], ENCODERS[V3_BASE]
    narrow = DeviceBudget(slices=20, size=224, patch=base.patch, dim=small.dim).needed()
    wide = DeviceBudget(slices=20, size=224, patch=base.patch, dim=base.dim).needed()
    assert wide == pytest.approx(2 * (narrow - RESIDENT_GB) + RESIDENT_GB, abs=0.02), (
        "activation bytes go linearly with width, so 768 costs twice 384 above the resident intercept"
    )
    assert wide > 12.68, "the derivation must clear the cap that OOM'd, or the fix forbids the same recipe"

    patch14 = DeviceBudget(slices=20, size=224, dim=base.dim).needed()
    assert patch14 > wide, "224 px is 256 tokens at patch 14 and 196 at 16, so a pixel-only cap overstates v3"


def test_needed_reads_the_batch():
    """The batch is a factor of the ask, and every arm at the settled batch must be untouched by it.

    MEASURED 2026-08-20: `base_dense` derived 56.3 GB against 30.3 free and parked for the whole
    contention bound without starting a step. Lowering `--batch` to 6 derived 56.3 AGAIN — the one lever
    a caller has against an overstated ask moved nothing, because batch was baked into `GB_PER_SLICE`
    as a constant. That did not make the larger-encoder axis expensive, it made it unreachable.
    """
    settled = DeviceBudget(slices=20).needed()
    assert DeviceBudget(slices=20, batch=FITTED_BATCH).needed() == settled, "the fit's own batch is the default"

    halved = DeviceBudget(slices=20, batch=FITTED_BATCH // 2).needed()
    assert halved == pytest.approx((settled - RESIDENT_GB) / 2 + RESIDENT_GB, abs=0.02), (
        "activation bytes go linearly with batch above the intercept, where the weights are held once"
    )

    base = ENCODERS[V3_BASE]
    wide = DeviceBudget(slices=20, patch=base.patch, dim=base.dim, batch=FITTED_BATCH).needed()
    narrower_step = DeviceBudget(slices=20, patch=base.patch, dim=base.dim, batch=FITTED_BATCH // 2).needed()
    assert narrower_step < wide, "a wider encoder must be reachable by asking for fewer studies at once"


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
