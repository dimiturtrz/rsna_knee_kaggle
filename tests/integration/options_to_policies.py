"""A policy a run ASKED FOR must be attached, not merely recorded — the failure class with a correct config.

`argv_to_checkpoint` proves the run's switches reach the manifest. That is one half. The other half is
`TrainingOptions.fitted_trainer` and `fitted_model`, which are eight hand-written `if self.flag:
object.method()` lines: a switch parsed, recorded and never attached trains the OLD recipe under the
NEW name, and every check downstream reads the config and agrees with itself. The report-supervision
no-op was exactly this — a weight with no vectors skipped the attachment silently, and the arm would
have been read as a refutation of the mechanism rather than of the invocation.

So this walks the POLICY METHODS of `Trainer` and `StudyModel` instead of naming today's six, and asks
that a run with every flag on call each one. A policy added tomorrow and never wired fails here, at the
commit that adds it, rather than on a 70-minute arm that reports a plausible number.
"""

from pathlib import Path

import pytest
import torch

from kneemri.models.backbone import Encoder
from kneemri.models.model import StudyDesign, StudyModel
from kneemri.training import train as train_module
from kneemri.training.train import TrainingOptions
from kneemri.training.trainer import Trainer

from ..unit.kneemri.models.model import DIM, SLOTS, _FakeBackbone

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
    "plan",
    "predict",
    "save_partial",
}
"""`Trainer` methods the FITTING calls, not the setup — they run per batch or per epoch, so no run
chooses them and no builder attaches them."""

NOT_A_WRAPPER = {
    "fingerprint",
    "forward",
    "from_config",
    "load_trained",
    "parameter_groups",
    "per_slice",
    "slice_features",
    "slot_features",
    "supervised",
}
"""`StudyModel` methods that are the model's own behaviour rather than something a run switches on."""

TRAINER_REPORTER = {"policies"}
MODEL_REPORTER = {"applied"}
"""Each object's account of WHICH policies landed on it — read by `Landed` before the card is spent.

They are the check this file makes at commit time, made again at run time, so they are the one kind of
public method a builder must NOT call: attaching them would mean the builder answering its own question.
The stale-exemption assert wants names the owner really has, so the two are named apart.
"""


def _options() -> TrainingOptions:
    """A run asking for EVERY optional policy at once, which is what makes one missing wire visible."""
    return TrainingOptions(
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
    monkeypatch.setattr(train_module.ReportSupervision, "from_file", lambda *arguments: object())
    policies = _policies(Trainer, THE_LOOP | TRAINER_REPORTER)
    called = _recorded(monkeypatch, Trainer, policies)

    model = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign())
    _options().fitted_trainer(model, DEVICE, ["study"], training_positions=16)

    assert set(policies) == called, f"{sorted(set(policies) - called)} is a policy no run can turn on"


def test_every_model_wrapper_a_run_can_ask_for_is_applied(monkeypatch: pytest.MonkeyPatch):
    """The model half of the same claim: adapters, checkpointing and compilation are run switches too."""
    monkeypatch.setattr(train_module.AutoModel, "from_pretrained", lambda repository: _FakeBackbone())
    wrappers = _policies(StudyModel, NOT_A_WRAPPER | MODEL_REPORTER)
    called = _recorded(monkeypatch, StudyModel, wrappers)

    # A CUDA device, though nothing is moved onto one: `fitted_model` compiles only for a card, so a CPU
    # device here would leave `compile_encoder` uncalled and read as a wrapper no run can turn on.
    _options().fitted_model(ENCODER, SLOTS, StudyDesign(), torch.device("cuda"))

    assert set(wrappers) == called, f"{sorted(set(wrappers) - called)} is a wrapper no run can turn on"
