"""The deployment chain: a saved checkpoint becomes a submission table.

This layer exists because every expensive failure in this project has been a JOINT between two units
that were each fully tested on their own.

- `augment` was written, unit-tested, and never called by `fit_epoch`. Vulture saw it used because its
  own test used it; the mirror gate saw it tested. Every gate checked the callee and none checked the
  caller.
- Extracting `Submission` was right on every local measure and broke the next kernel run, because the
  kit carrying the source was not rebuilt.
- Adding the anatomical prior added one parameter to the head, and every checkpoint written before it —
  including the five on the leaderboard — stopped loading under a strict load.

None of those is visible from inside a unit. All three are visible from here: save a member, load it the
way the kernel loads it, predict, combine, and check the table that would be submitted.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import override

import numpy as np
import torch
from torch import nn

from kneemri.backbone import ENCODERS, SMALL
from kneemri.ensemble import Ensemble
from kneemri.holdout_predictions import GROUP, HoldoutPredictions, Window
from kneemri.member_record import MemberRecord
from kneemri.members import MemberPackage
from kneemri.model import TARGETS, StudyDesign, StudyModel
from kneemri.train import TrainingOptions

DIM = 12
SLOTS = 6
SIZE = 14
STUDIES = 5
SEED = 2026
DIM_SMALL = 384
"""What ENCODERS says small is — asserted rather than imported, so a change to it fails here."""


class _FakeBackbone(nn.Module):
    """A stand-in encoder whose tokens vary across the feature dimension.

    Varying per dimension is load-bearing: an earlier double returned the same scalar in every feature,
    which the head's opening LayerNorm removes exactly, so the model's output did not depend on its
    pixels and no test using it could detect anything.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(2)])

    @override
    def forward(self, pixel_values: torch.Tensor) -> object:
        rows = pixel_values.mean(dim=(1, 3))[:, :DIM]
        tokens = rows[:, None, :].expand(-1, 5, -1) * torch.linspace(0.5, 1.5, DIM)
        return type("Output", (), {"last_hidden_state": tokens})()


def _images() -> tuple[np.ndarray, np.ndarray]:
    """A small corpus of studies with every slot present."""
    generator = torch.Generator().manual_seed(SEED)
    images = torch.randint(0, 256, (STUDIES, SLOTS, GROUP + 3, SIZE, SIZE), generator=generator, dtype=torch.uint8)
    return images.numpy(), torch.ones(STUDIES, SLOTS).numpy()


def _save(tmp_path: Path, design: StudyDesign, name: str) -> tuple[Path, StudyModel]:
    """Write a checkpoint through the ASSEMBLER a run writes with, and return it with its model.

    This used to hand-write three keys under a docstring claiming it was the shape `train.py` writes.
    It was not — the real save goes through `MemberRecord.payload`, which also stamps the fingerprint,
    its config, the kept epoch and the witness — and a hand-written copy of a format can only ever prove
    that the copy agrees with itself. The failure it hides is the one this file exists for: a change to
    what a checkpoint carries leaves this chain green while every consumer of a real member breaks.
    """
    torch.manual_seed(SEED)
    model = StudyModel(_FakeBackbone(), DIM, SLOTS, design).eval()
    options = TrainingOptions(
        slice_pooling=design.slice_pooling,
        anatomical_prior=design.anatomical_prior,
        image_size=SIZE,
        slices=GROUP + 3,
    )
    encoder = replace(ENCODERS[SMALL], dim=DIM)
    record = MemberRecord(options, encoder, model, torch.device("cpu"), [f"slot{index}" for index in range(SLOTS)])
    path = tmp_path / name
    torch.save(record.payload(model.state_dict(), holdout=0.83, epoch=0), path)
    return path, model


def test_a_saved_member_reloads_and_predicts_the_same(tmp_path: Path):
    """Save, reload the way an analysis or kernel does, and get the same probabilities back.

    The equality is the point. A state dict loads whenever the shapes line up, and shapes line up across
    a changed pooling rule, a changed normalisation or a changed slot count — none of which raise.
    """
    path, saved = _save(tmp_path, StudyDesign(), "member.pt")
    images, mask = _images()

    predictions = HoldoutPredictions(torch.device("cpu"))
    reloaded = predictions.load(path, _FakeBackbone(), DIM, SLOTS)

    direct = HoldoutPredictions(torch.device("cpu")).predict(
        saved, images, mask, Window(starts=[0], width=GROUP, size=SIZE)
    )
    through_disk = predictions.predict(reloaded, images, mask, Window(starts=[0], width=GROUP, size=SIZE))
    assert np.allclose(direct, through_disk, atol=1e-6)


def test_a_checkpoint_without_the_prior_still_loads(tmp_path: Path):
    """Every member on the leaderboard predates `head.prior`, and must survive its addition.

    Written as an integration test rather than a unit one because the failure was a joint: the head
    gained a parameter, and the thing that broke was the loader in a different module.
    """
    path, _ = _save(tmp_path, StudyDesign(), "old.pt")
    older = torch.load(path, map_location="cpu", weights_only=True)
    older["model"] = {name: value for name, value in older["model"].items() if name != "head.prior"}
    torch.save(older, path)

    model = HoldoutPredictions(torch.device("cpu")).load(path, _FakeBackbone(), DIM, SLOTS)
    assert float(model.head.prior.detach().abs().sum()) == 0.0


def test_members_combine_into_a_submission_table(tmp_path: Path):
    """Two members, rank-averaged, become one row per study and one column per target."""
    images, mask = _images()
    predictions = HoldoutPredictions(torch.device("cpu"))
    per_member = []
    for index, design in enumerate((StudyDesign(), StudyDesign(slice_pooling="max"))):
        path, _ = _save(tmp_path, design, f"member{index}.pt")
        model = predictions.load(path, _FakeBackbone(), DIM, SLOTS)
        per_member.append(predictions.predict(model, images, mask, Window(starts=[0, 1], width=GROUP, size=SIZE)))

    combined = Ensemble.combine(per_member)
    identifiers = [f"study{index}" for index in range(STUDIES)]
    table = Ensemble.submission(identifiers, combined)

    assert list(table) == ["StudyInstanceUID", *TARGETS]
    assert len(table["StudyInstanceUID"]) == STUDIES
    assert all(len(table[target]) == STUDIES for target in TARGETS)


def test_the_pooling_rule_travels_with_the_checkpoint(tmp_path: Path):
    """A member fitted with per-slice pooling must not be rebuilt as a channel-stacked one.

    They are different functions of the same weights, so loading under the wrong rule produces a model
    that runs, returns a submission, and is wrong in a way nothing in the run reveals.
    """
    path, _ = _save(tmp_path, StudyDesign(slice_pooling="max"), "max.pt")
    saved = json.loads(json.dumps(torch.load(path, map_location="cpu", weights_only=True)["config"]))
    assert saved["slice_pooling"] == "max"

    model = HoldoutPredictions(torch.device("cpu")).load(path, _FakeBackbone(), DIM, SLOTS)
    assert model.slice_pooling == "max"


def test_a_packaged_member_exposes_what_the_kernel_asks_it_for(tmp_path: Path):
    """The kernel path, which the tests above do not touch — and which was broken when this was written.

    The kernel picks a DINOv2 mount with `ENCODERS[member.variant]`. `variant` was a key in the manifest
    and in the MemberConfig type, and was NOT a property on Member, so that line was an AttributeError
    waiting for the next kernel run. Nothing else here could see it: the other tests reach the model
    through HoldoutPredictions, and the kernel reaches it through MemberPackage.

    A manifest written before variants existed must still answer, because every member published so far
    predates them.
    """
    torch.manual_seed(SEED)
    model = StudyModel(_FakeBackbone(), DIM, SLOTS).eval()
    torch.save(model.state_dict(), tmp_path / "m_a.pt")
    config = {
        "img": SIZE,
        "slices": GROUP + 3,
        "group": GROUP,
        "crop_mm": 130.0,
        "band": [0.2, 0.8],
        "slots": [f"slot{index}" for index in range(SLOTS)],
        "pool": "cls_mean",
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": 2,
                "members": [
                    {
                        "id": "a",
                        "file": "m_a.pt",
                        "fold": 0,
                        "config": config,
                        "fingerprint": model.fingerprint(SIZE, GROUP, torch.device("cpu")).tolist(),
                        "holdout": 0.82,
                        "annot": float("nan"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    member = MemberPackage(tmp_path).members[0]
    assert member.variant == SMALL
    assert ENCODERS[member.variant].dim == DIM_SMALL
