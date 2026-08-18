"""Whether a built model shows the switches its run asked for, on CPU-sized objects."""

from dataclasses import replace

import torch
from torch import nn

from kneemri.corpus.slot_reader import GROUP
from kneemri.models.model import ADAPTERS, COMPILED_ENCODER, MAXIMUM, RECOMPUTING, StudyDesign, StudyModel
from kneemri.training.landed import Landed, Request
from kneemri.training.objective import PAIR_POOL, PER_LABEL, RANK_TERM
from kneemri.training.remembered_ranking import RememberedRanking
from kneemri.training.trainer import AUTOCAST, AVERAGING, READ_SIZE, SCHEDULED, WINDOW, Trainer

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
    recompute=False,
    compiling=False,
    bf16=False,
    whole_stack=False,
    slices=24,
    train_size=0,
    ema=False,
    one_cycle=False,
    reports=False,
    rank_weight=0.0,
    pair_memory=0,
    per_label_loss=False,
    normalise_by_count=False,
    intensity_aug=False,
)
"""A run that asked for nothing — the recipe every number in the ledger was measured under.

Tests name only the axis they are about, via `replace`, so a new field does not have to be spelled at
every call site and the override stays type-checked rather than an untyped `**kwargs` bag.
"""


def _pair(model: StudyModel) -> Landed:
    return Landed(model, Trainer(model, torch.device("cpu")))


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

    assert len(wrong) == 7, "every switch asked for is absent from a model nothing was applied to"
    assert any(line.startswith(f"{ADAPTERS}: asked 8, the model shows 0") for line in wrong)

    model.low_rank(8)
    model.recompute_activations()
    landed.trainer.bfloat16()
    landed.trainer.whole_stack(24)
    landed.trainer.resample_to(392)
    landed.trainer.exponential_average()
    landed.trainer.one_cycle(10)

    assert landed.missing(wanted) == [], "a rank asks for adapters, not for a particular count of them"


def test_missing_loss() -> None:
    """A ranking arm whose objective is plain cross-entropy — the loss half of the same failure.

    The chain of `if`s in `TrainingOptions.objective` has a working objective on every branch, so a
    `--rank-weight` that never reached the constructor trains the plain recipe under the arm's name.
    """
    model = _model()
    plain = Landed(model, Trainer(model, torch.device("cpu")))
    ranked = replace(PLAIN, rank_weight=0.3, pair_memory=8, per_label_loss=True)

    wrong = plain.missing(ranked)

    assert [line.split(":")[0] for line in wrong] == [RANK_TERM, PAIR_POOL, PER_LABEL]

    remembered = Landed(model, Trainer(model, torch.device("cpu"), objective=RememberedRanking(0.3, capacity=8)))

    assert remembered.missing(ranked) == [f"{PER_LABEL}: asked True, the model shows False"], (
        "the memory reports the pool it really holds, and the per-label flag it was not given"
    )


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
