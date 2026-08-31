"""Whether a built model shows the switches its run asked for, on CPU-sized objects."""

from dataclasses import fields, replace

import numpy as np
import torch
from torch import nn

from core.data.corpus.slot_reader import GROUP
from core.data.labels.targets import (
    ANNOTATIONS,
    CALIBRATED,
    CONFIDENCE_FLOOR,
    WEIGHT_FLOOR,
    Supervision,
    Targets,
)
from kneemri.models.model import (
    ADAPTERS,
    COMPILED_ENCODER,
    DEPTH_REGIONS,
    NO_LAYER_DECAY,
    OPENED_BLOCKS,
    RECOMPUTING,
    StudyDesign,
    StudyModel,
)
from kneemri.models.slice_pooling import ATTEND, CHANNELS, MAXIMUM
from kneemri.training.gradient_clip import UNCLIPPED
from kneemri.training.landed import NOT_A_SWITCH, Landed, Request
from kneemri.training.mixup import OFF
from kneemri.training.objective import PAIR_POOL, PER_LABEL, RANK_TERM
from kneemri.training.options import TrainingOptions
from kneemri.training.remembered_ranking import RememberedRanking
from kneemri.training.trainer import (
    AUTOCAST,
    AVERAGING,
    CLIPPING,
    LAYER_DECAY,
    ONE_CYCLE_WARMUP,
    OPTIMISER_DECAY,
    READ_SIZE,
    SCHEDULED,
    WARMUP,
    WEIGHT_DECAY,
    WINDOW,
    Adaptation,
    Trainer,
)

DIM = 8
SLOTS = 6


class _FakeBlock(nn.Module):
    """A transformer block's SHAPE — Linears held by a container, not a bare Linear.

    `low_rank` rebinds a Linear on its parent, so a block that IS a Linear has no child to rebind and
    adapts nothing. The structure is the thing under test, so the double has to have it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.Linear(DIM, DIM)
        self.mlp = nn.Linear(DIM, DIM)


class _FakeBackbone(nn.Module):
    """Enough encoder to be adapted and checkpointed, and none of the weights.

    `gradient_checkpointing_enable` is HF's, so the double implements it the way transformers does —
    by flipping the PUBLIC `is_gradient_checkpointing` flag. That is the whole point of the key: the
    real backbone is the authority on whether checkpointing is on, and a check reading our own request
    would agree with itself on an encoder that ignored the call.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_FakeBlock() for _ in range(4)])
        self.is_gradient_checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, object] | None = None) -> None:
        del gradient_checkpointing_kwargs
        self.is_gradient_checkpointing = True


PLAIN = Request(
    lora_rank=0,
    slice_regions=1,
    recompute=False,
    compiling=False,
    bf16=False,
    whole_stack=False,
    slices=24,
    train_size=0,
    ema=False,
    one_cycle=False,
    accumulate=1,
    reports=False,
    rank_weight=0.0,
    distill_weight=0.0,
    feature_weight=0.0,
    dissent_weight=0.0,
    contrast_weight=0.0,
    auc_margin=0.0,
    teaching=False,
    teaching_features=False,
    pair_memory=0,
    per_label_loss=False,
    normalise_by_count=False,
    intensity_aug=False,
    weight_floor=False,
    calibrated_targets=False,
    finding_calibration=False,
    severity_weight=False,
    hold_annotations=False,
    grade_ladder=False,
    side_weight=0.0,
    focus_label="",
    # FOUR, not the trainer's default six: `_FakeBackbone` holds four blocks, so opening "the last
    # six" of it opens four, and the request that matches an untouched encoder is the one that names
    # what is there. The mismatch a real encoder would show is the point of the strict comparison.
    unfreeze_last=4,
    layer_decay=NO_LAYER_DECAY,
    mixup=OFF,
    weight_decay=WEIGHT_DECAY,
    warmup=ONE_CYCLE_WARMUP,
    grad_clip=UNCLIPPED,
)
"""A run that asked for nothing — the recipe every number in the ledger was measured under.

Tests name only the axis they are about, via `replace`, so a new field does not have to be spelled at
every call site and the override stays type-checked rather than an untyped `**kwargs` bag.
"""


def _targets(supervision: Supervision | None = None) -> Targets:
    """A table of the smallest shape that carries a supervision — no file, no labels, no pandas."""
    scores = np.full((2, 12), 0.5, dtype=np.float32)
    return Targets(scores, np.ones_like(scores), ["a", "b"], supervision)


def _pair(model: StudyModel) -> Landed:
    return Landed(model, Trainer(model, torch.device("cpu")), _targets())


def _model() -> StudyModel:
    """Per-slice pooling, because the whole-stack window is one of the switches being checked and the
    trainer refuses it on channel stacking."""
    return StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling=MAXIMUM))


def test_observed() -> None:
    """The effects come off the objects: no adapters and fp16 before, both changed after."""
    model = _model()
    before = _pair(model).observed()

    assert before[ADAPTERS] == 0
    assert before[RECOMPUTING] is False
    assert before[COMPILED_ENCODER] is False
    assert before[AUTOCAST] == torch.float16
    assert before[WINDOW] == GROUP
    assert before[READ_SIZE] is None
    assert before[AVERAGING] is False
    assert before[SCHEDULED] is False

    landed = _pair(model)
    model.low_rank(4)
    model.recompute_activations()
    landed.trainer.bfloat16()
    landed.trainer.whole_stack(24)
    landed.trainer.resample_to(392)
    landed.trainer.exponential_average()
    landed.trainer.one_cycle(10)
    after = landed.observed()

    assert after[ADAPTERS] == 8, "one adapter per Linear in each of the backbone's blocks"
    assert after[RECOMPUTING] is True
    assert after[AUTOCAST] == torch.bfloat16
    assert after[WINDOW] == 24
    assert after[READ_SIZE] == 392
    assert after[AVERAGING] is True, "the averaged weights are a subclass, not a flag on the live ones"
    assert after[SCHEDULED] is True


def test_missing() -> None:
    """Silence when every switch landed, and the un-landed one named with both sides.

    The LoRA case is the one that has actually happened: the flag parses, the config records rank 8,
    and an encoder whose projections the matcher did not recognise wraps nothing.
    """
    model = _model()
    landed = _pair(model)
    assert landed.missing(PLAIN) == []

    wanted = replace(
        PLAIN, lora_rank=8, recompute=True, bf16=True, whole_stack=True, train_size=392, ema=True, one_cycle=True
    )
    wrong = landed.missing(wanted)

    assert len(wrong) == 8, "every switch asked for is absent from a model nothing was applied to"
    assert any(line.startswith(f"{ADAPTERS}: asked 8, the model shows 0") for line in wrong)
    assert any(line.startswith(f"{OPENED_BLOCKS}: asked 0, the model shows 4") for line in wrong), (
        "the depth is the EIGHTH: a LoRA ask means no open base block, and this encoder has four — "
        "the same reading that says the adapters never landed, from the other side"
    )

    model.low_rank(8)
    model.recompute_activations()
    landed.trainer.bfloat16()
    landed.trainer.whole_stack(24)
    landed.trainer.resample_to(392)
    landed.trainer.exponential_average()
    landed.trainer.one_cycle(10)

    assert landed.missing(wanted) == [], "a rank asks for adapters, not for a particular count of them"


def test_agrees() -> None:
    """A float that survived a round trip through arithmetic landed; one that missed by its size did not.

    The exact case that cost a queue slot: `band_dense_llrd` asked for a 0.7 layer rate decay and the
    trainer recovered 0.7000000000000004 from its per-layer rates, which an equality test called unlanded.
    """
    landed = _pair(_model())

    assert landed.agrees(LAYER_DECAY, 0.7000000000000004, 0.7) is True, "the last bits of a multiply"
    assert landed.agrees(LAYER_DECAY, 0.75, 0.7) is False, "a switch that did not land misses by its size"
    assert landed.agrees(LAYER_DECAY, 1.0, 0.7) is False, "the default, which is what unlanded looks like"
    assert landed.agrees(ADAPTERS, 12, 8) is True, "a rank asks for adapters, not a count of them"
    assert landed.agrees(ADAPTERS, 0, 8) is False
    assert landed.agrees(WINDOW, 24, 24) is True, "an int still compares exactly"
    assert landed.agrees(AUTOCAST, torch.bfloat16, torch.bfloat16) is True


def test_missing_loss() -> None:
    """A ranking arm whose objective is plain cross-entropy — the loss half of the same failure.

    The chain of `if`s in `TrainingOptions.objective` has a working objective on every branch, so a
    `--rank-weight` that never reached the constructor trains the plain recipe under the arm's name.
    """
    model = _model()
    plain = Landed(model, Trainer(model, torch.device("cpu")), _targets())
    ranked = replace(PLAIN, rank_weight=0.3, pair_memory=8, per_label_loss=True)

    wrong = plain.missing(ranked)

    assert [line.split(":")[0] for line in wrong] == [RANK_TERM, PAIR_POOL, PER_LABEL]

    remembered = Landed(
        model, Trainer(model, torch.device("cpu"), objective=RememberedRanking(0.3, capacity=8)), _targets()
    )

    assert remembered.missing(ranked) == [f"{PER_LABEL}: asked True, the model shows False"], (
        "the memory reports the pool it really holds, and the per-label flag it was not given"
    )


def test_missing_labels() -> None:
    """The two switches that map to a DATA argument instead of to a policy object.

    `Targets.pair` and `Targets.load` both default their supervision, so a `main` that stopped passing
    the run's own value would fit the plain table under a calibrated arm's name and every recorded
    config would still agree with itself. The floor is compared as its VALUE, since that is what the
    loss reads and what a broken bool-to-value rule would get wrong.
    """
    model = _model()
    asked = replace(PLAIN, weight_floor=True, calibrated_targets=True)
    trainer = Trainer(model, torch.device("cpu"))

    unsupervised = Landed(model, trainer, _targets()).missing(asked)

    assert [line.split(":")[0] for line in unsupervised] == [WEIGHT_FLOOR, CALIBRATED]
    assert unsupervised[0] == f"{WEIGHT_FLOOR}: asked {CONFIDENCE_FLOOR!r}, the model shows 0.0"

    supervised = Supervision(floor=CONFIDENCE_FLOOR, calibrated=True)

    assert Landed(model, trainer, _targets(supervised)).missing(asked) == []


def test_missing_held_annotations() -> None:
    """The withheld annotations, which is the switch whose silent no-op would cost the WHOLE instrument.

    `band_dense_hold` exists only to produce predictions that never saw the 58 image-read annotations. A
    `--hold-annotations` that never reached `Supervision` trains the ordinary table, scores within the
    floor of its control — exactly the pre-registered read — and hands `supervision_gap` a member as
    tainted as every other. Nothing downstream would say a word, which is why it is compared here.
    """
    model = _model()
    trainer = Trainer(model, torch.device("cpu"))
    asked = replace(PLAIN, hold_annotations=True)

    assert Landed(model, trainer, _targets()).missing(asked) == [f"{ANNOTATIONS}: asked False, the model shows True"], (
        "a table built with the annotations under an arm that asked to withhold them"
    )
    assert Landed(model, trainer, _targets(Supervision(annotations=False))).missing(asked) == []


def test_missing_depth() -> None:
    """The depth axis, which had no check at all because LoRA legitimately ignores the flag.

    A naive observed-vs-asked over open blocks false-positives on every adaptation arm, so the ask is
    mapped through the mode instead — and what is left is a strict equality that catches the zero case
    the slice arithmetic used to invert.
    """
    model = _model()
    trainer = Trainer(model, torch.device("cpu"), adaptation=Adaptation(unfreeze_last=2))

    assert Landed(model, trainer, _targets()).missing(replace(PLAIN, unfreeze_last=2)) == []
    assert Landed(model, trainer, _targets()).missing(PLAIN) == [f"{OPENED_BLOCKS}: asked 4, the model shows 2"], (
        "a run that opened two blocks under a four-block recipe is the arm nobody ran"
    )

    frozen = _model()
    head_only = Trainer(frozen, torch.device("cpu"), adaptation=Adaptation(unfreeze_last=0))

    assert Landed(frozen, head_only, _targets()).missing(replace(PLAIN, unfreeze_last=0)) == []


def test_missing_depth_regions() -> None:
    """THE FLAG THAT SPLIT NOTHING FOR ITS WHOLE LIFE. `slot_features` reaches `over_depth` only when no
    slice aggregator was built, so `--slice-regions 4 --slice-pooling attend` parsed, was recorded in the
    manifest and trained — splitting nothing, with no error and no log line. Two fusion arms had to be
    run separately because of it and the ledger carried a region count no run ever used.

    The ask is the flag and the answer is the count `over_depth` will actually split on, which is what
    turns a silent misrecord into a refusal before the card is spent.
    """
    regions = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling=MAXIMUM, slice_regions=4))

    assert _pair(regions).missing(replace(PLAIN, slice_regions=4)) == []

    attended = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling=ATTEND, slice_regions=4))

    assert _pair(attended).missing(replace(PLAIN, slice_regions=4)) == [
        f"{DEPTH_REGIONS}: asked 4, the model shows 1"
    ], "attention owns the slice axis, so the regions are inert and the run must not be allowed to start"

    stacked = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling=CHANNELS, slice_regions=4))

    assert _pair(stacked).missing(replace(PLAIN, slice_regions=4)) != [], (
        "channel stacking removes the slice axis entirely, which is the same hole from the other side"
    )


def test_not_a_switch() -> None:
    """Every training option is compared here or written off here — the partition, asserted.

    THE HOLE THIS CLOSES IS THE NEXT KNOB, not a current one. `Request` names seventeen of forty-eight
    options and nothing forced the other thirty-one to be decided, so a switch added to the trainer
    could reach the card with no check at all — which is the same shape as the five switches that were
    landed-checked, stamped and never RUN until `switch_coverage` made that partition total too.

    Both directions, because a stale write-off is the quieter failure: a name that leaves
    `TrainingOptions` and stays here reads as a considered decision about a flag that no longer exists.
    """
    options = {field.name for field in fields(TrainingOptions)}
    compared = {field.name for field in fields(Request)} & options

    assert options - compared - set(NOT_A_SWITCH) == set(), (
        "a training option that neither Landed compares nor NOT_A_SWITCH writes off — decide which, "
        "because a switch nothing asks the model about is forty minutes of card under a recipe nobody ran"
    )
    assert set(NOT_A_SWITCH) - options == set(), "a write-off for a flag TrainingOptions no longer has"
    assert compared & set(NOT_A_SWITCH) == set(), "a switch cannot be both compared and written off"


def test_wanted() -> None:
    """The request is translated once — `--whole-stack` decides the window, and 0 read size means None."""
    stack = replace(PLAIN, compiling=True, bf16=True, whole_stack=True, slices=32).wanted()

    assert stack[WINDOW] == 32
    assert stack[READ_SIZE] is None
    assert stack[AUTOCAST] == torch.bfloat16
    assert stack[COMPILED_ENCODER] is True

    grouped = replace(PLAIN, lora_rank=4, slices=32, train_size=420).wanted()

    assert grouped[WINDOW] == GROUP, "the slice count is inert unless the run asked for the whole stack"
    assert grouped[READ_SIZE] == 420
    assert grouped[AUTOCAST] == torch.float16
    assert grouped[OPENED_BLOCKS] == 0, (
        "under adaptation `parameter_groups` ignores the depth deliberately, so the honest ask is no "
        "open base blocks — asking through the flag would fail every LoRA arm"
    )
    assert replace(PLAIN, unfreeze_last=2).wanted()[OPENED_BLOCKS] == 2

    # The three optimiser-path knobs pass through unchanged — no translation to own, and the default
    # must be the inherited value, since these are the first flags whose OFF state is a number every
    # ledger row was measured at rather than a policy nobody attached.
    assert PLAIN.wanted()[OPTIMISER_DECAY] == WEIGHT_DECAY
    assert PLAIN.wanted()[WARMUP] == ONE_CYCLE_WARMUP
    assert PLAIN.wanted()[CLIPPING] == UNCLIPPED
    tuned = replace(PLAIN, weight_decay=0.1, warmup=0.05, grad_clip=1.0).wanted()
    assert (tuned[OPTIMISER_DECAY], tuned[WARMUP], tuned[CLIPPING]) == (0.1, 0.05, 1.0)
