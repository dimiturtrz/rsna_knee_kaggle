from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from torch import Tensor

from core.data.corpus.slot_reader import GROUP
from core.data.corpus.study_cache import CachedCorpus
from core.data.labels.targets import Targets
from core.findings import TARGETS
from kneemri.models.drop_path import NO_DROP
from kneemri.models.model import NO_LAYER_DECAY, StudyDesign, StudyModel
from kneemri.models.slot_head import REPORT_DIMS
from kneemri.training.adaptation import WEIGHT_DECAY, Adaptation
from kneemri.training.augmentation import (
    INTENSITY_AUGMENT,
    PADDING_MODE,
    PATCH_ERASE,
    SCALE_JITTER,
    SLOT_DROPOUT,
    Augmentation,
    AugmentConfig,
)
from kneemri.training.feature_distillation import FeatureDistilledObjective, Widths
from kneemri.training.ghosting import GHOSTING
from kneemri.training.grades import GRADE_LADDER, Predicted
from kneemri.training.gradient_clip import UNCLIPPED
from kneemri.training.label_focus import FOCUS, UNFOCUSED
from kneemri.training.loss_quantiles import LOSS_QUANTILES
from kneemri.training.mixup import OFF, UNMIXED
from kneemri.training.objective import (
    CONTRAST_TERM,
    COUNT_NORMALISED,
    DISSENT_TERM,
    DISTILL_TERM,
    FEATURE_TERM,
    MARGIN_TERM,
    PAIR_POOL,
    PER_LABEL,
    RANK_TERM,
    Objective,
    RankedObjective,
)
from kneemri.training.patch_erasing import PatchErasing
from kneemri.training.progress_schedule import ProgressSchedule
from kneemri.training.report_supervision import ReportSupervision
from kneemri.training.sampling import Disagreed
from kneemri.training.sharpness import FLAT
from kneemri.training.side_ranking import SIDE_TERM
from kneemri.training.teacher_features import FEATURE_TEACHER, TeacherFeatures
from kneemri.training.teacher_targets import TEACHER, TeacherTargets
from kneemri.training.trainer import (
    ACCUMULATION,
    AUG_SCHEDULE,
    AUTOCAST,
    AVERAGING,
    BACKBONE_RATE,
    CERTAINTY_RAMP,
    CLIPPING,
    CORPUS_OVERSAMPLE,
    DROP_PATH,
    DROPPED_BLOCKS,
    HARD_SCHEDULE,
    LAYER_DECAY,
    MIXUP,
    ONE_CYCLE_WARMUP,
    OPTIMISER_DECAY,
    READ_SIZE,
    REPORTS,
    SAMPLING,
    SAMPLING_RAMP,
    SCHEDULED,
    SHARPNESS,
    TARGET_CORPUS,
    WARMUP,
    WINDOW,
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


def test_blend():
    """Attaching the blend is visible in the trainer's own account of itself.

    This is the POSITIVE WITNESS for `--mixup`. Everything else about the flag is silent: a value that
    never reached the trainer leaves the loop drawing an identity blend, trains perfectly well, and
    reports the un-mixed recipe's number under the mixed arm's name — the failure shape `landed` exists
    for, and one this project has already paid for three times.
    """
    trainer = _trainer()
    assert trainer.policies()[MIXUP] == OFF, "the default must be the loop every ledger number was measured under"
    trainer.blend(0.2)
    assert trainer.policies()[MIXUP] == pytest.approx(0.2)


def test_draw_with():
    """Attaching the draw policy is visible in the trainer's own account of itself, and defaults to off.

    The POSITIVE WITNESS for `--sample-spread`: an unattached policy is `Uniform`, which returns the
    positions untouched and does not even advance the generator, so a spread file that never reached the
    trainer trains exactly the ledger's uniform run and reports it under the sampled arm's name.
    """
    trainer = _trainer()
    assert trainer.policies()[SAMPLING] == OFF, "every ledger number was measured on a uniform draw"
    trainer.draw_with(Disagreed(np.linspace(0.01, 0.3, 8), alpha=0.5))
    assert trainer.policies()[SAMPLING] == pytest.approx(0.5)


def test_erase_with():
    """Attaching the patch maps is visible in the trainer's own account of itself, and defaults to off.

    The POSITIVE WITNESS for `--patch-erase`: an unattached erasing holds a zero rate and returns the
    pixels untouched, so a rate that never reached the trainer trains exactly the ledger's unerased run
    and reports it under the erased arm's name.
    """
    trainer = _trainer()
    assert trainer.policies()[PATCH_ERASE] == OFF, "every ledger number was measured without erasing"
    trainer.erase_with(PatchErasing(np.zeros((1, 4, 4)), 0.25))
    assert trainer.policies()[PATCH_ERASE] == pytest.approx(0.25)


def test_clip_gradients():
    """Attaching the ceiling is visible in the trainer's own account of itself, and defaults to off.

    The POSITIVE WITNESS for `--grad-clip`: an unattached clip is a `GradientClip(0.0)` that returns
    before it touches a gradient, so a value that never reached the trainer trains exactly the ledger's
    unbounded run and reports it under the clipped arm's name.
    """
    trainer = _trainer()
    assert trainer.policies()[CLIPPING] == UNCLIPPED, "every ledger number was measured unbounded"
    trainer.clip_gradients(1.0)
    assert trainer.policies()[CLIPPING] == pytest.approx(1.0)


def test_sharpen():
    """The radius is visible in the trainer's own account of itself, and it REFUSES accumulation.

    The POSITIVE WITNESS for `--sharpness`: a radius that never reached the trainer trains exactly the
    ledger's ordinary step and reports it under the SAM arm's name. The refusal is the second half —
    the ascent is built from the whole group's gradient while the second forward covers only the last
    micro-batch, and a silent approximation would read as a badly chosen radius rather than as a bug.
    """
    trainer = _trainer()
    assert trainer.policies()[SHARPNESS] == FLAT, "every ledger number was measured on the plain step"
    trainer.sharpen(0.05)
    assert trainer.policies()[SHARPNESS] == pytest.approx(0.05)
    trainer.accumulate(2)
    with pytest.raises(ValueError, match="accumulate 1"):
        trainer.sharpen(0.05)


class _Droppable(torch.nn.Module):
    """The shape the walk looks for — a block that hangs its residual policy on `drop_path`."""

    def __init__(self) -> None:
        super().__init__()
        self.drop_path = torch.nn.Identity()


def _droppable() -> torch.nn.Module:
    return _Droppable()


def test_drop_paths():
    """The rate AND the block count are both reported, because only the count says the rate landed.

    A stand-in trunk carries no `drop_path` attribute, so this also pins the refusal: a rate that
    reaches nothing must raise rather than train an ordinary run under the arm's name.
    """
    trainer = _trainer()
    assert trainer.policies()[DROP_PATH] == NO_DROP, "every ledger number ran on the intact trunk"
    assert trainer.policies()[DROPPED_BLOCKS] == 0
    trainer.model.backbone.blocks = torch.nn.ModuleList([_droppable(), _droppable()])
    trainer.drop_paths(0.1)
    assert trainer.policies()[DROP_PATH] == pytest.approx(0.1)
    assert trainer.policies()[DROPPED_BLOCKS] == 2


def test_warm_up():
    """The fraction is what the schedule is BUILT with, so the witness is the rate the cycle starts at.

    Reading the field back would only prove the setter assigns. `one_cycle` reads it once, at
    construction, which is why `run_setup` attaches this first — a warmup set afterwards would leave the
    run annealing on 0.15 while its recipe recorded the value asked for.

    THE WITNESS IS THE RATE PART-WAY IN, not the rate at step zero: OneCycleLR starts every cycle at
    `max_lr / div_factor` whatever the fraction is, so an initial-rate comparison passes on a warmup that
    never landed. Four steps into forty, a 0.05 warmup has already peaked and turned down while a 0.5
    warmup is a fifth of the way up.
    """
    trainer = _trainer()
    assert trainer.policies()[WARMUP] == pytest.approx(ONE_CYCLE_WARMUP), "the inherited default"

    trainer.warm_up(0.5)
    assert trainer.policies()[WARMUP] == pytest.approx(0.5)
    slow = _rate_after(trainer, warmup_steps=4, total_steps=40)
    brief = _trainer()
    brief.warm_up(0.05)
    assert all(
        climbing < peaked
        for climbing, peaked in zip(slow, _rate_after(brief, warmup_steps=4, total_steps=40), strict=True)
    ), "a longer warmup must still be climbing where a brief one has already turned down"


def test_shortest_cycle():
    """A one-epoch smoke under a quarter warmup is four steps, and OneCycleLR divided by zero on it."""
    trainer = _trainer()
    trainer.warm_up(0.25)
    assert trainer.shortest_cycle() == 5
    rates = _rate_after(trainer, warmup_steps=4, total_steps=4)
    assert all(rate > 0 for rate in rates)


def _rate_after(trainer: Trainer, warmup_steps: int, total_steps: int) -> list[float]:
    """The group rates a few steps into a freshly attached cycle."""
    trainer.one_cycle(total_steps=total_steps)
    assert trainer.schedule is not None
    for _ in range(warmup_steps):
        trainer.schedule.step()
    return [group["lr"] for group in trainer.optimiser.param_groups]


def test_weight_decay():
    """MEASURED off the optimiser's own group, not off the `Adaptation` that was passed in.

    A decay is a keyword to `AdamW`'s constructor, so the way it fails is silent by construction: drop
    the keyword and every group carries the library default while the config records the value asked
    for. Reading the built optimiser back is what makes that a refusal before the card is spent.
    """
    assert _trainer().witness.weight_decay() == pytest.approx(WEIGHT_DECAY), "the inherited default"

    tighter = Trainer(
        StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max")),
        torch.device("cpu"),
        Adaptation(weight_decay=0.1),
    )
    assert tighter.witness.weight_decay() == pytest.approx(0.1)
    assert tighter.policies()[OPTIMISER_DECAY] == pytest.approx(0.1), "and it is what the pre-flight check reads"


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


def test_teach_from():
    """Attaching readings flips the LANDING WITNESS, which is a different fact from the term's weight.

    A distilling run carries two numbers that can disagree: the objective's `--distill-weight`, which is
    a float a run can hold while the array behind it never loaded, and this — whether an ensemble's
    readings actually reached the trainer. The check before the card compares both, so the second one
    has to be observable off the trainer rather than inferred from the first.
    """
    trainer = Trainer(
        StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max")),
        torch.device("cpu"),
    )
    assert trainer.policies()[TEACHER] is False

    scores = np.zeros((2, len(TARGETS)), dtype=np.float32)
    trainer.teach_from(TeacherTargets(scores))

    assert trainer.policies()[TEACHER] is True
    assert trainer.teacher is not None
    assert trainer.teacher.rows(np.array([1])).shape == (1, len(TARGETS))


def test_teach_features_from():
    """A SEPARATE witness from `teach_from`, because a run may carry one teacher file, the other, or both.

    Merged into one switch, the check before the card would answer "a teacher arrived" for the arm whose
    teacher did not — and a feature arm whose npz joined on nothing descends the plain recipe under the
    distilling arm's name, which is the whole failure this pair of witnesses exists against.
    """
    width = 5
    trainer = Trainer(
        StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max")),
        torch.device("cpu"),
    )
    assert trainer.policies()[FEATURE_TEACHER] is False
    assert trainer.policies()[TEACHER] is False

    trainer.teach_features_from(TeacherFeatures(np.zeros((2, width), dtype=np.float32)))

    assert trainer.policies()[FEATURE_TEACHER] is True
    assert trainer.policies()[TEACHER] is False, "the two witnesses are independent"
    assert trainer.feature_teacher is not None
    assert trainer.feature_teacher.rows(np.array([1])).shape == (1, width)


def test_optimiser_steps_a_parameter_the_objective_owns():
    """The projection a feature objective holds must be IN the optimiser, not merely reachable.

    `parameter_groups` collects the head by exclusion from the backbone, so a tensor owned by the LOSS
    is in no group unless it is handed over at construction — and the optimiser is built once, in
    `Trainer.__init__`, so there is no later place to repair it. Unattached it would stay at its
    initialisation for the whole arm and the cosine would report a fact about a random map (knee-7gyw).
    """
    objective = FeatureDistilledObjective(Widths(DIM, 7))
    trainer = Trainer(
        StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max")),
        torch.device("cpu"),
        objective=objective,
    )

    stepped = {id(parameter) for group in trainer.optimiser.param_groups for parameter in group["params"]}
    assert id(objective.projection.weight) in stepped


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
        DISTILL_TERM: 0.0,
        FEATURE_TERM: 0.0,
        DISSENT_TERM: 0.0,
        CONTRAST_TERM: 0.0,
        MARGIN_TERM: 0.0,
        PAIR_POOL: 0,
        PER_LABEL: False,
        COUNT_NORMALISED: False,
        GRADE_LADDER: False,
        SIDE_TERM: 0.0,
        FOCUS: "",
        UNFOCUSED: 0.0,
        INTENSITY_AUGMENT: False,
        SLOT_DROPOUT: 0.0,
        PATCH_ERASE: 0.0,
        GHOSTING: 0.0,
        SCALE_JITTER: 0.08,
        PADDING_MODE: "border",
        # The three optimiser-path knobs, in `constructed` because they are the same on both sides of
        # the attachments below: nothing `one_cycle`/`whole_stack`/`exponential_average` does may move
        # them, and a warmup that changed when the schedule was attached would be the drift the landed
        # check exists to catch.
        OPTIMISER_DECAY: WEIGHT_DECAY,
        WARMUP: ONE_CYCLE_WARMUP,
        CERTAINTY_RAMP: 0,
        CLIPPING: UNCLIPPED,
        CORPUS_OVERSAMPLE: 0.0,
        TARGET_CORPUS: "",
    }

    assert trainer.policies() == {
        **constructed,
        AUTOCAST: torch.float16,
        WINDOW: GROUP,
        READ_SIZE: None,
        AVERAGING: False,
        SCHEDULED: False,
        REPORTS: False,
        TEACHER: False,
        FEATURE_TEACHER: False,
        ACCUMULATION: 1,
        LAYER_DECAY: NO_LAYER_DECAY,
        BACKBONE_RATE: Adaptation().backbone_lr,
        MIXUP: OFF,
        SAMPLING: OFF,
        SAMPLING_RAMP: 0,
        SHARPNESS: FLAT,
        DROP_PATH: NO_DROP,
        DROPPED_BLOCKS: 0,
        AUG_SCHEDULE: (1.0, 1.0, 1.0, 0.0, 0.0),
        HARD_SCHEDULE: (1.0, 1.0, 1.0, 0.0, 0.0),
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
        TEACHER: False,
        FEATURE_TEACHER: False,
        ACCUMULATION: 1,
        LAYER_DECAY: NO_LAYER_DECAY,
        BACKBONE_RATE: Adaptation().backbone_lr,
        MIXUP: OFF,
        SAMPLING: OFF,
        SAMPLING_RAMP: 0,
        SHARPNESS: FLAT,
        DROP_PATH: NO_DROP,
        DROPPED_BLOCKS: 0,
        AUG_SCHEDULE: (1.0, 1.0, 1.0, 0.0, 0.0),
        HARD_SCHEDULE: (1.0, 1.0, 1.0, 0.0, 0.0),
    }


def test_encoder_rate_decay():
    """The ratio is MEASURED off the optimiser's groups, so a flag that never landed cannot report itself.

    Flat has no pair for a ratio to be defined between and must answer `NO_LAYER_DECAY` rather than
    raising on a one-element slice — that is the branch every arm in the ledger takes.
    """
    model = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max"))
    flat = Trainer(model, torch.device("cpu"), Adaptation(unfreeze_last=4))
    assert flat.witness.encoder_rate_decay() == NO_LAYER_DECAY

    decayed = Trainer(
        StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max")),
        torch.device("cpu"),
        Adaptation(unfreeze_last=4, layer_decay=0.5),
    )
    assert decayed.witness.encoder_rate_decay() == pytest.approx(0.5)
    assert decayed.policies()[LAYER_DECAY] == pytest.approx(0.5), "and it is what the pre-flight check reads"


def test_backbone_rate():
    """The ANCHOR is measured, and the decayed case proves the ratio alone could never have caught it.

    Both trainers below ask for the same rate and land the same anchor; only the decay differs. That is
    the point — `encoder_rate_decay` reports 0.5 for the second and `NO_LAYER_DECAY` for the first while
    neither says anything about the rate being divided, which is how a run can floor its loss against its
    control with every recorded flag correct (knee-62sol).
    """
    asked = 9.8e-5

    def trainer_at(**adaptation: float | int) -> Trainer:
        model = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max"))
        return Trainer(model, torch.device("cpu"), Adaptation(backbone_lr=asked, **adaptation))

    flat = trainer_at(unfreeze_last=4)
    assert flat.witness.backbone_rate() == pytest.approx(asked)
    assert flat.policies()[BACKBONE_RATE] == pytest.approx(asked), "and it is what the pre-flight check reads"

    decayed = trainer_at(unfreeze_last=4, layer_decay=0.5)
    assert decayed.witness.backbone_rate() == pytest.approx(asked), "the decay divides the blocks BELOW the anchor"

    frozen = trainer_at(unfreeze_last=0)
    assert frozen.witness.backbone_rate() == pytest.approx(asked), "the group is built at the rate even with none open"
    assert frozen.witness.encoder_rate_decay() == NO_LAYER_DECAY, "so the anchor is defined where the ratio is not"


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


def test_partial(tmp_path: Path):
    """The name is derived in one place, so the writer and the resumer cannot disagree about it."""
    assert Trainer.partial(tmp_path / "member_fold0.pt") == tmp_path / "member_fold0.partial"


def test_restore_weights(tmp_path: Path):
    """The weights go back into the model and the reading that made them a member comes back with them."""
    trainer = _trainer()
    destination = tmp_path / "member_fold0.pt"
    state = {key: value.detach().cpu() for key, value in trainer.model.head.state_dict().items()}
    trainer.save_partial(destination, {"model": trainer.model.state_dict(), "holdout": 0.81, "epoch": 7})

    for tensor in trainer.model.head.parameters():
        torch.nn.init.zeros_(tensor)
    held = trainer.restore_weights(destination)

    assert held["epoch"] == 7
    assert torch.equal(trainer.model.head.state_dict()[next(iter(state))], state[next(iter(state))])


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


def test_fit_epoch_leaves_loss_quantiles():
    """An epoch leaves the per-study loss distribution behind it, keyed the same way every time.

    The keys are fixed rather than derived so two epochs are the same instrument read twice; a fresh
    trainer has nothing to report and must still answer with the full key set.
    """
    trainer = _trainer()
    assert trainer.trace.quantiles == {}, "nothing has been fitted, so there is no distribution yet"

    trainer.fit_epoch(_corpus(), _targets(), np.arange(6, dtype=np.int64), 3, np.random.default_rng(SEED))

    quantiles = trainer.trace.quantiles
    assert tuple(quantiles) == LOSS_QUANTILES
    assert all(np.isfinite(level) for level in quantiles.values())
    assert quantiles["q25"] <= quantiles["q50"] <= quantiles["q75"] <= quantiles["q90"] <= quantiles["q95"]


def test_ramp_certainty():
    """The curriculum blends cell weights from the floor toward full trust WITHOUT changing their mass.

    The mass is the point: the objective divides by the weight sum, so a ramp that shrank the total
    would change the loss scale and make the arm incomparable to its un-ramped control rather than
    testing a curriculum.
    """
    trainer = _trainer()
    weights = np.array([[1.0, 0.5, 0.25]], dtype=np.float32)
    assert np.array_equal(trainer._ramped_weights(weights, 0), weights), "off by default"

    trainer.ramp_certainty(4)
    assert trainer.certainty_ramp == 4

    first = trainer._ramped_weights(weights, 0)
    assert not np.array_equal(first, weights), "the first epoch reads every cell as equally uncertain"
    assert first.sum() == pytest.approx(weights.sum(), rel=1e-5), "the ramp moves weight, it does not remove it"
    assert first.std() < weights.std(), "flattened toward the floor before it is flattened back out"

    assert np.allclose(trainer._ramped_weights(weights, 4), weights, atol=1e-6), "full trust at the ramp's end"
    assert np.allclose(trainer._ramped_weights(weights, 99), weights, atol=1e-6), "and it never overshoots"


def test_harden():
    """Hardness is the same progress schedule as augmentation, applied to label trust and the draw's tilt.

    Classes: the default schedule (the ledger's recipe, weights returned as the very same array), a
    schedule at zero (every weighted cell at the floor, held cells still zero), a schedule mid-ramp, and
    the legacy whole-epoch ramp expressed as a schedule (equal to float rounding at every epoch boundary).
    """
    trainer = _trainer()
    weights = np.array([[1.0, 0.5, 0.25, 0.0]], dtype=np.float32)
    assert trainer.policies()[HARD_SCHEDULE] == (1.0, 1.0, 1.0, 0.0, 0.0)
    assert trainer._ramped_weights(weights, 0.0) is weights, "the default touches nothing"
    assert trainer._harden(0.0) == 1.0

    trainer.pace(
        4, strength=ProgressSchedule(), hardness=ProgressSchedule(start=0.0, mid=1.0, end=0.5, ramp=0.5, tail=0.25)
    )
    assert trainer.policies()[HARD_SCHEDULE] == (0.0, 1.0, 0.5, 0.5, 0.25)
    assert trainer._harden(0.0) == 0.0
    assert trainer._harden(1.0) == pytest.approx(0.5)
    assert trainer._harden(2.0) == pytest.approx(1.0)
    assert trainer._harden(4.0) == pytest.approx(0.5)

    floored = trainer._ramped_weights(weights, 0.0)
    assert floored[0, 3] == 0.0, "a held cell is not an uncertain one and must stay out of the loss"
    assert np.allclose(floored[0, :3], floored[0, 0]), "at zero trust every weighted cell pulls equally"
    assert floored.sum() == pytest.approx(weights.sum(), rel=1e-5)
    assert np.allclose(trainer._ramped_weights(weights, 2.0), weights, atol=1e-6)

    legacy = _trainer()
    legacy.ramp_certainty(4)
    scheduled = _trainer()
    scheduled.pace(4, strength=ProgressSchedule(), hardness=ProgressSchedule(start=0.0, mid=1.0, end=1.0, ramp=1.0))
    for epoch in range(5):
        assert np.allclose(
            legacy._ramped_weights(weights, epoch), scheduled._ramped_weights(weights, epoch), atol=1e-6
        ), "the schedule reproduces the superseded ramp at each epoch it was defined on"


def test_harden_leaves_the_draw_unchanged_by_default():
    """The default schedule draws exactly today's studies under the same seed; zero strength draws flat."""
    spread = np.linspace(0.01, 0.30, 40)
    positions = np.arange(40, dtype=np.int64)
    today = Disagreed(spread, alpha=0.5).drawn(positions, np.random.default_rng(7))
    trainer = _trainer()
    trainer.draw_with(Disagreed(spread, alpha=0.5))
    trainer._harden(0.0)
    assert np.array_equal(trainer.sampling.drawn(positions, np.random.default_rng(7)), today)

    trainer.pace(4, strength=ProgressSchedule(), hardness=ProgressSchedule(start=0.0, ramp=0.5))
    trainer._harden(0.0)
    assert trainer.sampling.alpha == 0.0
    assert trainer.sampling.tilt() == 0.5, "the switch check still reads the configured tilt"


def test_pace():
    """One strength schedule drives every consumer; the default reports the constant schedule the ledger ran."""
    trainer = _trainer()
    assert trainer.policies()[AUG_SCHEDULE] == (1.0, 1.0, 1.0, 0.0, 0.0)
    trainer.augmentation = Augmentation(AugmentConfig(slot_dropout=0.4))
    trainer.blend(0.2)
    trainer.pace(
        4, strength=ProgressSchedule(start=0.0, mid=1.0, end=0.0, ramp=0.5, tail=0.5), hardness=ProgressSchedule()
    )
    assert trainer.policies()[AUG_SCHEDULE] == (0.0, 1.0, 0.0, 0.5, 0.5)
    assert trainer._strengthen(0) == 0.0
    assert trainer.augmentation.config.slot_dropout == 0.0
    assert trainer.mixup.draw(3, np.random.default_rng(SEED)).weight == UNMIXED
    assert trainer._strengthen(2) == pytest.approx(1.0)
    assert trainer.augmentation.config.slot_dropout == pytest.approx(0.4)
    assert trainer.policies()[SLOT_DROPOUT] == pytest.approx(0.4), "the landed check reads the asked-for base"
