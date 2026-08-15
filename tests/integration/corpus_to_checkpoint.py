"""The training chain: a corpus becomes a checkpoint that INFERENCE reads the way training wrote it.

`checkpoint_to_submission` starts where this ends — from a checkpoint that already exists — so the
half of the pipeline that PRODUCES one had no integration cover at all. That is the half where this
project's expensive failures live: nine times a value was recorded in the config while a consumer read
its own default, and every one of them loaded cleanly, fingerprinted cleanly, and predicted from
something the weights had never seen.

`comp/witness_check.py` is the check that closes that gap, and it can only run against a real packaged
member on a real cache — outside CI, by hand, after a 70-minute run. This makes the same claim on a
synthetic member in three seconds: fit, save, package, load the way a kernel loads, and re-derive the
witness through `Ensemble.predict_member`. If the training and serving paths ever disagree about how a
member is read, this fails here rather than on a submission slot.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from kneemri.backbone import Encoder
from kneemri.ensemble import Ensemble
from kneemri.fitting import Fitting, Fold, RunPlan
from kneemri.member_record import MemberRecord
from kneemri.members import MemberPackage
from kneemri.model import TARGETS, StudyDesign, StudyModel
from kneemri.study_cache import CachedCorpus
from kneemri.targets import Targets
from kneemri.trainer import Trainer

from ..unit.kneemri.member_record import _Options
from ..unit.kneemri.model import DIM, SLOTS, _FakeBackbone

SEED = 2026
STUDIES = 8
SLICES = 6
SIZE = 14
DEVICE = torch.device("cpu")
WITNESS_TOLERANCE = 1e-5
"""How far the two paths may differ before this is a reading defect rather than float32 rounding."""


def _fold() -> Fold:
    generator = np.random.default_rng(SEED)
    corpus = CachedCorpus(
        generator.integers(0, 256, (STUDIES, SLOTS, SLICES, SIZE, SIZE), dtype=np.uint8),
        np.ones((STUDIES, SLOTS), dtype=np.float32),
    )
    scores = np.zeros((STUDIES, len(TARGETS)), dtype=np.float32)
    scores[STUDIES // 2 :, :] = 1.0
    targets = Targets(scores, np.ones_like(scores), [f"s{index}" for index in range(STUDIES)])
    positions = np.arange(STUDIES, dtype=np.int64)
    return Fold(corpus=corpus, targets=targets, evaluation=targets, training=positions, holdout=positions)


@dataclass(frozen=True)
class _Checkpoint:
    """A written checkpoint, read back the way every consumer of one reads it."""

    payload: dict[str, object]

    @property
    def config(self) -> dict[str, object]:
        return cast(dict[str, object], self.payload["config"])

    @property
    def witness(self) -> dict[str, object]:
        return cast(dict[str, object], self.config["witness"])

    @property
    def studies(self) -> list[str]:
        return [str(name) for name in cast(list[object], self.witness["studies"])]

    @property
    def predictions(self) -> np.ndarray:
        return np.asarray(self.witness["predictions"], dtype=np.float64)


def _trained(destination: Path) -> _Checkpoint:
    """Fit a member the way `train.main` does, and return the checkpoint it would write."""
    torch.manual_seed(SEED)
    # Per-slice pooling, which is what whole-stack reading is defined over.
    design = StudyDesign(slice_pooling="max")
    model = StudyModel(_FakeBackbone(), DIM, SLOTS, design)
    trainer = Trainer(model, DEVICE)
    # Whole-stack, because that is what every arm on the leaderboard reads — and because a member
    # fitted whole and served three slices at a time is the exact defect this chain exists to catch.
    trainer.whole_stack(SLICES)
    options = _Options(image_size=SIZE, slices=SLICES, whole_stack=True)
    record = MemberRecord(
        options,
        Encoder(variant="small", repository="facebook/dinov2-small", dim=DIM),
        model,
        DEVICE,
        [f"S{index}" for index in range(SLOTS)],
    )
    kept = Fitting(trainer, record, RunPlan(destination=destination, epochs=1, batch=4, seed=SEED)).run(_fold())
    payload = record.payload(kept.state, kept.holdout, kept.epoch, kept.witness)
    torch.save(payload, destination)
    return _Checkpoint(payload)


def _packaged(tmp_path: Path, path: Path, checkpoint: _Checkpoint) -> MemberPackage:
    """The manifest `package_members` writes, so the member is read through the mounted path."""
    manifest = {
        "format": 2,
        "members": [
            {
                "id": "fold0",
                "file": path.name,
                "fold": 0,
                "config": checkpoint.config,
                "fingerprint": np.asarray(checkpoint.payload["fingerprint"]).tolist(),
                "holdout": float(cast(float, checkpoint.payload["holdout"])),
                "annot": float("nan"),
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return MemberPackage(tmp_path)


def test_a_fitted_member_is_read_by_inference_the_way_training_wrote_it(tmp_path: Path):
    """The witness, re-derived through `Ensemble.predict_member` — `witness_check`'s claim, in CI.

    The fingerprint cannot make this claim: it pins image size and group and nothing else, so a member
    served under a different slot list, decode resolution, pooling rule or slice depth passes it while
    predicting from pixels it was never fitted on.
    """
    path = tmp_path / "member_fold0.pt"
    checkpoint = _trained(path)
    package = _packaged(tmp_path, path, checkpoint)
    member = package.members[0]

    model = package.load(member, _FakeBackbone(), DIM, DEVICE)
    corpus = _fold().corpus
    positions = np.array([int(name[1:]) for name in checkpoint.studies], dtype=np.int64)
    width = member.slices if member.whole_stack else member.group
    start = (SLICES - width) // 2
    images = torch.from_numpy(np.ascontiguousarray(corpus.images[positions, :, start : start + width]))
    mask = torch.from_numpy(corpus.mask[positions])

    served = Ensemble(package, {member.identifier: model}, DEVICE).predict_member(member, images, mask, overlap=True)

    assert np.allclose(served, checkpoint.predictions, atol=WITNESS_TOLERANCE), (
        "inference reads this member differently than the training path that produced its weights"
    )


def test_the_witness_moves_when_inference_reads_the_member_wrongly(tmp_path: Path):
    """The check above is only worth its runtime if a WRONG reading fails it — so break one.

    Dropping `whole_stack` from the config is defect knee-y8y exactly: a member fitted on its whole
    stack, served three slices at a time. It loads, it fingerprints, and it reached the leaderboard.
    """
    path = tmp_path / "member_fold0.pt"
    checkpoint = _trained(path)
    positions = np.array([int(name[1:]) for name in checkpoint.studies], dtype=np.int64)
    checkpoint.config["whole_stack"] = False
    package = _packaged(tmp_path, path, checkpoint)
    member = package.members[0]

    model = package.load(member, _FakeBackbone(), DIM, DEVICE)
    corpus = _fold().corpus
    images = torch.from_numpy(np.ascontiguousarray(corpus.images[positions]))
    mask = torch.from_numpy(corpus.mask[positions])

    served = Ensemble(package, {member.identifier: model}, DEVICE).predict_member(member, images, mask, overlap=True)

    difference = float(np.abs(served - checkpoint.predictions).max())
    assert difference > WITNESS_TOLERANCE, (
        f"a windowed reading of a whole-stack member moved nothing ({difference:.3g}) — "
        "the witness cannot see the defect it exists for"
    )


def test_the_checkpoint_carries_the_reading_it_was_fitted_under(tmp_path: Path):
    """A member is loadable only if it recorded HOW it was read — the nine-defect failure class.

    Asserted through `Member`'s accessors rather than the raw dict, because the defect was never a
    missing key: every one of these was written faithfully and read by nobody.
    """
    path = tmp_path / "member_fold0.pt"
    member = _packaged(tmp_path, path, _trained(path)).members[0]

    assert member.whole_stack, "a whole-stack member served in windows is a different function"
    assert member.slices == SLICES
    assert member.read_size == SIZE
    assert member.slot_count == SLOTS
    assert member.variant == "small"


def test_a_killed_run_leaves_a_member_that_can_still_be_packaged(tmp_path: Path):
    """The partial is the SAME payload, so a run stopped at its best epoch is submittable.

    A 0.8413 member was once left as weights with no fingerprint and no recorded reading, which
    `MemberPackage` refuses — and the point of killing that run early had been to keep it.
    """
    path = tmp_path / "member_fold0.pt"
    _trained(path)
    kept = _Checkpoint(torch.load(path.with_suffix(".partial"), map_location="cpu", weights_only=False))
    torch.save(kept.payload, path)

    package = _packaged(tmp_path, path, kept)
    model = package.load(package.members[0], _FakeBackbone(), DIM, DEVICE)

    assert model is not None, "a partial carries everything a mounted package needs to run it"
