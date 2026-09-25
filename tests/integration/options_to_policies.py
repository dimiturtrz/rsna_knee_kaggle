"""A policy a run ASKED FOR must be attached, not merely recorded — the failure class with a correct config.

`argv_to_checkpoint` proves the run's switches reach the manifest. That is one half. The other half is
`RunSetup.trainer` and `RunSetup.model`, which are eight hand-written `if self.flag:
object.method()` lines: a switch parsed, recorded and never attached trains the OLD recipe under the
NEW name, and every check downstream reads the config and agrees with itself. The report-supervision
no-op was exactly this — a weight with no vectors skipped the attachment silently, and the arm would
have been read as a refutation of the mechanism rather than of the invocation.

So this walks the POLICY METHODS of `Trainer` and `StudyModel` instead of naming today's six, and asks
that a run with every flag on call each one. A policy added tomorrow and never wired fails here, at the
commit that adds it, rather than on a 70-minute arm that reports a plausible number.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from kneemri.models.encoder import Encoder
from kneemri.models.model import StudyDesign, StudyModel
from kneemri.models.slice_pooling import SLICE_POOLINGS
from kneemri.models.slot_head import SLOT_POOLINGS
from kneemri.training import run_setup as build_module
from kneemri.training.config_tree import TrainConfig
from kneemri.training.run_setup import RunSetup
from kneemri.training.trainer import Trainer

from ..unit.kneemri.models.model import DIM, SLOTS, _BlockedBackbone, _FakeBackbone

SLICES = 6
SIZE = 14
TRAIN_SIZE = 28
DEVICE = torch.device("cpu")
ENCODER = Encoder("fake", "fake/encoder", DIM)

THE_LOOP = {
    "augment_bag",
    "batches",
    "evaluated",
    "fit_epoch",
    "partial",
    "plan",
    "predict",
    "restore_weights",
    "save_partial",
    "shortest_cycle",
}
"""`Trainer` methods the FITTING calls, not the setup — they run per batch or per epoch, so no run
chooses them and no builder attaches them.

`shortest_cycle` is the odd one: it is arithmetic, not a phase. It returns the fewest steps whose warm
phase OneCycleLR will not divide by zero on, and its ONLY caller is `one_cycle` — which is itself a
policy, so this test replaces it with a recorder and its body never runs. The helper therefore reads as
a policy no run can turn on while being reachable from every one_cycle run there is. Exempt on the same
ground as the rest of this set: nothing attaches it, because it is not a thing to attach."""

NOT_A_WRAPPER = {
    "check_whole_stack",
    "context_width",
    "fingerprint",
    "folded",
    "forward",
    "from_config",
    "load_trained",
    "over_depth",
    "parameter_groups",
    "per_slice",
    "slice_features",
    "slot_features",
    "supervised",
    "train",
    "window",
}
"""`StudyModel` methods that are the model's own behaviour rather than something a run switches on.

`window` is arithmetic over the encoder's depth, asked by whoever is about to build the optimiser. It
attaches nothing and changes nothing; the run switch it serves is `--unfreeze-last`/`--unfreeze-offset`,
and what LANDED is checked through `frozen_top_blocks`, not by anyone having called this.

`check_whole_stack` is a REFUSAL, not a wrapper: `--whole-stack` is switched on by the trainer, and the
model's part is to say at startup whether it can take the window at all. It attaches nothing, so a
builder calling it would be attaching a policy that does not exist.

`context_width` is a MEASUREMENT for the same reason: the feature term has to be built at the student's
real head width, and asking the model is what keeps that from being a guess. It switches nothing on.

`train` is an OVERRIDE of `nn.Module.train`, and it is here because it is the opposite of a switch. A
run cannot ask for it and no builder attaches it: it exists because `nn.Module.train()` recurses into
every child and would re-enable the BatchNorm statistics that freezing a trunk is supposed to hold
still, so the model re-pins them on the way out. A wrapper a run turns on is a choice; this one fires
on every call whether anything asked or not, which is the point — the drift it prevents cost one whole
trunk-family arm before it existed.

`folded` is the slice axis being collapsed, and the aggregator it delegates to is the choice — `slices`
when a run asked for one, `over_depth` otherwise, and neither when the design is not per-slice. It is
the SITE of that choice, not the switch: it appeared as a public method when `slot_features` learned to
skip absent slots (5561bd4a), because both the masked and the unmasked path collapse the same way and
one of them had to name it. A run cannot ask for it and a builder attaching it would attach nothing —
same standing as `over_depth` and `slice_features` beside it."""

TRAINER_REPORTER = {"policies"}
MODEL_REPORTER = {"applied"}
"""Each object's account of WHICH policies landed on it — read by `Landed` before the card is spent.

They are the check this file makes at commit time, made again at run time, so they are the one kind of
public method a builder must NOT call: attaching them would mean the builder answering its own question.
The stale-exemption assert wants names the owner really has, so the two are named apart.

`encoder_rate_decay`, `weight_decay` and `backbone_rate` were exempted here until 2026-09-12, on the
argument that each is a WITNESS rather than a switch: all three flags are consumed while the optimiser is
BUILT, out of `Adaptation`, so there is no method for a builder to call and nothing for it to attach, and
each reads its value back off the built optimiser's own groups — which is what makes it a witness, since
a value that never landed cannot report itself. That argument survives; what changed is where they live.
They are now `OptimiserWitness` in `kneemri.training.witness`, reached as `Trainer.witness`, so they are
no longer members of `Trainer` and naming them here fails the stale-exemption assert instead of the
policy sweep. An attribute is not callable, so the sweep never sees `witness` and needs no exemption for
it. If a witness ever moves back onto `Trainer`, it comes back into this set with it.
"""


def _options() -> TrainConfig:
    """A run asking for EVERY optional policy at once, which is what makes one missing wire visible."""
    return TrainConfig().with_leaves(
        slices=SLICES,
        image_size=SIZE,
        train_size=TRAIN_SIZE,
        whole_stack=True,
        slice_pooling="max",
        one_cycle=True,
        ema=True,
        bf16=True,
        lora_rank=4,
        recompute_activations=True,
        compile_encoder=True,
        report_weight=1.0,
        report_vectors=Path("reports.npz"),
        distill_weight=0.3,
        teacher_targets=Path("teacher.npz"),
        certainty_ramp=6,
        patch_maps=Path("patches.npz"),
        patch_erase=0.25,
    )


class _FakeFeatures:
    """A teacher-features stand-in that answers a WIDTH, which is the only thing the builder asks it."""

    @staticmethod
    def width() -> int:
        return DIM


def _features_arm() -> TrainConfig:
    """The same run with the REPRESENTATION teacher instead of the logit one, the terms it excludes off."""
    return _options().with_leaves(
        distill_weight=0.0,
        teacher_targets=None,
        feature_weight=1.0,
        teacher_features=Path("features.npz"),
    )


def _policies(owner: type, exempt: set[str]) -> list[str]:
    named = [
        name
        for name, value in vars(owner).items()
        if callable(value) and not name.startswith("_") and name not in exempt
    ]
    assert exempt < {name for name, value in vars(owner).items() if not name.startswith("_")}, (
        f"a name exempted on {owner.__name__} is not one of its members any more — stale exemption"
    )
    return named


def _recorded(monkeypatch: pytest.MonkeyPatch, owner: type, names: list[str]) -> set[str]:
    """Every listed method replaced by a recorder, so the REAL policy never has to be runnable here."""
    called: set[str] = set()

    def recorder(name: str):
        def record(self: object, *arguments: object, **keywords: object) -> None:
            called.add(name)

        return record

    for name in names:
        monkeypatch.setattr(owner, name, recorder(name))
    return called


def test_every_trainer_policy_a_run_can_ask_for_is_attached(monkeypatch: pytest.MonkeyPatch):
    """A switch parsed and stamped but never attached is a run that trains the recipe before it."""
    monkeypatch.setattr(build_module.ReportSupervision, "from_file", lambda *arguments: object())
    monkeypatch.setattr(build_module.TeacherTargets, "from_file", lambda *arguments: object())
    monkeypatch.setattr(build_module.TeacherFeatures, "from_file", lambda *arguments: _FakeFeatures())
    monkeypatch.setattr(build_module.PatchErasing, "loaded", lambda *arguments: object())
    policies = _policies(Trainer, THE_LOOP | TRAINER_REPORTER)
    called = _recorded(monkeypatch, Trainer, policies)

    # TWO runs, because the two distillations are exclusive by construction: `--feature-weight` refuses
    # to compose with `--distill-weight`, so one options object cannot reach both teachers and a single
    # arm would read the one it left out as a policy no run can turn on.
    model = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign())
    RunSetup(_options()).trainer(model, DEVICE, ["study"], np.arange(16))
    RunSetup(_features_arm()).trainer(model, DEVICE, ["study"], np.arange(16))

    assert set(policies) == called, f"{sorted(set(policies) - called)} is a policy no run can turn on"


def _slice_arm(name: str) -> tuple[TrainConfig, StudyDesign]:
    return _options().with_leaves(slice_pooling=name), StudyDesign(slice_pooling=name)


def _slot_arm(name: str) -> tuple[TrainConfig, StudyDesign]:
    return _options().with_leaves(slot_pooling=name), StudyDesign(slot_pooling=name)


READOUT_VOCABULARIES = {"slice_pooling": (SLICE_POOLINGS, _slice_arm), "slot_pooling": (SLOT_POOLINGS, _slot_arm)}
"""The design axes whose names build a module with WEIGHTS — the only ones that can fail silently.

Ranked deliberately (knee-29s5): `scheme`, `centre` and `tie` are vocabularies too, and the integration
suite pins one value of each the same way, but they decide WHICH PIXELS are read rather than what is
trained. A mis-build there changes a reading, which `argv_to_checkpoint`'s design ratchet already
catches; a mis-build here leaves a tensor that no optimiser moves, which nothing caught for two arms."""


@pytest.mark.parametrize(("axis", "vocabulary_and_arm"), READOUT_VOCABULARIES.items())
def test_every_readout_a_run_can_ask_for_reaches_the_optimiser(
    axis: str,
    vocabulary_and_arm: tuple[Mapping[str, object], Callable[[str], tuple[TrainConfig, StudyDesign]]],
    monkeypatch: pytest.MonkeyPatch,
):
    """A pooling a run can NAME must arrive as a module whose weights an optimiser will actually move.

    THE PAIR THIS EXISTS FOR is `options.py` -> `model.py`, and it is the failure that already happened
    (knee-7gyw): `--slice-pooling attend` parsed, stamped into the manifest, built the right module, and
    then trained none of it, because the head group was `self.head.parameters()` and the aggregator is a
    SIBLING of the head. The gate is zero-initialised on purpose so an untrained pooling is exactly the
    mean over slices — so nothing crashed, nothing scored badly, and two ledger entries measured the mean
    while claiming to measure attention.

    Walks the VOCABULARY rather than naming the poolings that exist today: a name added to the table and
    not wired into a param group fails at the commit that adds it. Asserted by exclusion — every
    trainable tensor the model owns is in some group — because naming the modules is what failed.
    """
    # The BLOCKED stand-in, because `_options()` asks for LoRA and adaptation refuses a backbone with no
    # Linear inside its blocks. Worth carrying rather than switching off: the adapters live INSIDE the
    # encoder, so they must land in the backbone group, and this assertion is over every group at once.
    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained", lambda repository, **_: (_BlockedBackbone(), {"missing_keys": []})
    )
    vocabulary, arm = vocabulary_and_arm
    for name in vocabulary:
        options, design = arm(name)
        model = RunSetup(options).model(ENCODER, SLOTS, design, DEVICE)

        groups = model.parameter_groups(head_lr=1e-3, backbone_lr=8e-6, window=model.window(4))
        grouped = [id(tensor) for group in groups for tensor in cast("list[torch.nn.Parameter]", group["params"])]
        trainable = {id(tensor) for tensor in model.parameters() if tensor.requires_grad}

        assert trainable <= set(grouped), f"{axis}={name} leaves a trainable tensor outside every group"
        assert len(grouped) == len(set(grouped)), f"{axis}={name} hands one tensor to two groups"


def test_every_model_wrapper_a_run_can_ask_for_is_applied(monkeypatch: pytest.MonkeyPatch):
    """The model half of the same claim: adapters, checkpointing and compilation are run switches too."""
    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained", lambda repository, **_: (_FakeBackbone(), {"missing_keys": []})
    )
    wrappers = _policies(StudyModel, NOT_A_WRAPPER | MODEL_REPORTER)
    called = _recorded(monkeypatch, StudyModel, wrappers)

    # A CUDA device, though nothing is moved onto one: `RunSetup.model` compiles only for a card, so a CPU
    # device here would leave `compile_encoder` uncalled and read as a wrapper no run can turn on.
    RunSetup(_options()).model(ENCODER, SLOTS, StudyDesign(), torch.device("cuda"))

    assert set(wrappers) == called, f"{sorted(set(wrappers) - called)} is a wrapper no run can turn on"
