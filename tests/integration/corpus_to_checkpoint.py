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

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast, override

import numpy as np
import torch

from kneemri.corpus.study_cache import CachedCorpus
from kneemri.findings import TARGETS
from kneemri.inference.ensemble import Ensemble
from kneemri.inference.members import MemberConfig, MemberPackage
from kneemri.kaggle.package_members import MemberManifest, PackagedMember
from kneemri.labels.targets import Targets
from kneemri.models.backbone import Encoder
from kneemri.models.model import StudyDesign, StudyModel
from kneemri.runs.member_record import MemberRecord
from kneemri.training.fitting import Fitting, Fold, RunPlan
from kneemri.training.options import TrainingOptions
from kneemri.training.run_setup import RunSetup

from ..unit.kneemri.models.model import DIM, SLOTS, _FakeBackbone

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
    # Whole-stack, because that is what every arm on the leaderboard reads — and because a member
    # fitted whole and served three slices at a time is the exact defect this chain exists to catch.
    # Attached through the SETUP method `train.main` calls: hand-wiring `Trainer(...).whole_stack(...)`
    # here shared no code with the run it claims to reproduce, so a policy that stopped being attached
    # in `main` would leave this chain green.
    options = TrainingOptions(image_size=SIZE, slices=SLICES, whole_stack=True, slice_pooling="max")
    trainer = RunSetup(options).trainer(model, DEVICE, [], STUDIES)
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
    """The manifest through the REAL producer, so this chain breaks when the shipped shape changes.

    Hand-writing the rows here would freeze a copy of the format: `package_members` could rename a key
    and this would keep passing against a package no kernel would ever be handed.
    """
    document = MemberManifest([PackagedMember(path, checkpoint.payload)]).document()
    (tmp_path / "manifest.json").write_text(document, encoding="utf-8")
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


class _WatchedConfig(dict[str, object]):
    """A member's config that records which keys the SERVE path actually looked at."""

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__(config)
        self.reads: set[str] = set()

    @override
    def __getitem__(self, key: str) -> object:
        self.reads.add(key)
        return super().__getitem__(key)

    @override
    def get(self, key: str, default: object = None) -> object:
        self.reads.add(key)
        return super().get(key, default)


PROVENANCE_ONLY = {
    "backbone_lr",
    "cache_studies",
    "ema",
    "head_lr",
    "img",
    "pair_memory",
    "rank_weight",
    "report_weight",
    "unfreeze_last",
    "witness",
}
"""Keys serving is right not to consult, each for its own reason.

The recipe half — the two rates, `ema`, `unfreeze_last`, `rank_weight`, `pair_memory`, `report_weight` —
describes how
the weights were FITTED, which no forward pass can act on; they exist so a member on the leaderboard can
be traced back to the arm that made it. `img` is read only when `train_size` is absent, which is every
member fitted before the read size could differ from the cache's. `witness` is `witness_check`'s to
re-derive. `cache_studies` names which CACHE the member was fitted on, and the kernel decodes DICOMs
rather than reading one — it is read by `Member.cache`, which is the local tools' path (the oracle, the
curve, the witness), so a smoke member re-read there gets its own 60 studies instead of the full 4407.
Everything else a run stamps — the crop, the centre, the band, the tie rule, the slot list,
the depth, the pooling, the precision — is asserted below to be read by the serve path itself.
"""


def test_every_key_a_member_stamps_is_read_when_it_is_served(tmp_path: Path):
    """A key written into the manifest and read by nobody is the nine-defect class, serve side.

    The training half of this claim is `argv_to_checkpoint`'s field walk. This is the other end: the
    kernel decides a decode from the members (scheme, depth, pixel groups) and then predicts, so every
    key a run stamps should be consulted by one of those steps or by a named tool. One that is consulted
    by neither travels the whole chain describing a reading that nothing performs.
    """
    path = tmp_path / "member_fold0.pt"
    checkpoint = _trained(path)
    package = _packaged(tmp_path, path, checkpoint)
    watched = _WatchedConfig(package.members[0].config)
    member = replace(package.members[0], config=cast(MemberConfig, watched))
    # Into the package, because the ensemble groups its reads from the members IT holds — a watched
    # member the ensemble never sees would report every key of the grouping step as dead.
    package.members[0] = member

    ensemble = Ensemble(package, {member.identifier: package.load(member, _FakeBackbone(), DIM, DEVICE)}, DEVICE)
    ensemble.reads()
    ensemble.slices([member])
    corpus = _fold().corpus
    positions = np.array([int(name[1:]) for name in checkpoint.studies], dtype=np.int64)
    images = torch.from_numpy(np.ascontiguousarray(corpus.images[positions]))
    ensemble.predict_member(member, images, torch.from_numpy(corpus.mask[positions]), overlap=True)

    assert set(watched) > PROVENANCE_ONLY, "a key exempted here is not stamped any more — stale exemption"
    unread = set(watched) - watched.reads - PROVENANCE_ONLY
    assert not unread, f"{sorted(unread)} is stamped on every member and read by no part of serving"
