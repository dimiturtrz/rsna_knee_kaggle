from dataclasses import replace
from pathlib import Path

import torch

from kneemri.corpus.crop_centre import CROP_MM
from kneemri.models.backbone import Encoder
from kneemri.runs.member_record import MemberRecord, StampedRun
from kneemri.training.train import TrainingOptions

from ..models.model import SLOTS, _FakeBackbone  # the stand-in encoder this suite already trusts

DIM = 8
OPTIONS = TrainingOptions(slice_pooling="max", whole_stack=True, backbone_lr=4e-5)
"""The REAL options object, because a stand-in copy of it is a copy that drifts.

This was 19 hand-written fields mirroring `TrainingOptions`. A field added to the run and read by
`MemberRecord` would have been recorded by every real run and by nothing here — and what this file
asserts is precisely that the checkpoint stamps the values the RUN used.
"""


def _record() -> MemberRecord:
    from kneemri.models.model import StudyModel  # noqa: PLC0415 - keeps the fake backbone import local

    torch.manual_seed(2026)
    model = StudyModel(_FakeBackbone(), DIM, SLOTS)
    encoder = Encoder(variant="small", repository="facebook/dinov2-small", dim=DIM)
    return MemberRecord(OPTIONS, encoder, model, torch.device("cpu"), [f"S{i}" for i in range(SLOTS)])


def test_recipe():
    """Both the checkpoint's `options` block and the metrics file read this, so it has one home.

    Iterated from the options object rather than listed: a field added to a run reaches the recipe
    without anyone remembering to add it here, which is the failure this project keeps paying for.
    """
    recipe = _record().recipe

    assert recipe.keys() == vars(OPTIONS).keys(), "every option, not a chosen few"
    assert recipe["backbone_lr"] == "4e-05", "as strings, because a recipe is read and diffed, not run"
    assert recipe["whole_stack"] == "True"


def test_payload():
    """A checkpoint carries the reading it was fitted under, not just its weights.

    The two saves used to disagree — the final one wrote config and fingerprint, the periodic one
    wrote neither — so a run killed at its best epoch left weights MemberPackage refuses. Every field
    asserted here is one a loader reads back to decide how the member must be run.
    """
    record = _record()
    payload = record.payload({"w": torch.zeros(1)}, holdout=0.8413, epoch=6)

    assert payload["holdout"] == 0.8413
    assert payload["epoch"] == 6
    config = payload["config"]
    assert config["slice_pooling"] == "max", "a member run under the other aggregation predicts from wrong pixels"
    assert config["whole_stack"] is True
    assert config["backbone_lr"] == 4e-5, "the value the run USED, never a module constant"
    # The objective's own fields, because a member fitted on cross-step pairs and one fitted on
    # batch-local pairs are otherwise indistinguishable on disk — and an arm that cannot prove which
    # loss produced it cannot be read against a control.
    assert config["rank_weight"] == OPTIONS.rank_weight
    assert config["pair_memory"] == OPTIONS.pair_memory
    assert payload["fingerprint"], "without this the member cannot be verified, so it cannot be submitted"
    assert payload["fingerprint_config"]["n_slot"] == SLOTS


def test_read_size():
    """The size the model SAW, which is the cache's unless the run resampled off it.

    Both classes matter and they fail in opposite directions. Zero must mean the cache's own size, or
    every member fitted before resampling gets a fingerprint taken at the wrong resolution. A set
    value must WIN over the cache's, or a member trained at 392 from a 448 cache advertises 448 and is
    rebuilt as a different function on different pixels - which is exactly what slice_pooling did.
    """
    from dataclasses import replace  # noqa: PLC0415 - local to the one test that needs it

    record = _record()
    assert record.read_size == 336, "train_size 0 means the model read the cache as decoded"
    assert record.payload({}, 0.8, 0)["config"]["train_size"] == 336

    resampled = replace(record, options=replace(OPTIONS, train_size=392))
    assert resampled.read_size == 392
    assert resampled.payload({}, 0.8, 0)["fingerprint_config"]["img"] == 392


def test_witness():
    """The witness names STUDIES, not corpus positions, and carries what training predicted for them.

    A position is meaningless without the same corpus index, and the entire point is to be checkable
    from another process against a cache built later — so the identifiers have to travel.
    """
    import numpy as np  # noqa: PLC0415 - only this test needs arrays

    record = _record()
    identifiers = [f"study-{index}" for index in range(20)]
    holdout = np.array([5, 7, 11], dtype=np.int64)
    predictions = np.arange(3 * 12, dtype=np.float64).reshape(3, 12) / 100.0

    witness = record.witness(identifiers, holdout, predictions)
    assert witness["studies"] == ["study-5", "study-7", "study-11"]
    assert len(witness["predictions"]) == 3
    assert record.payload({}, 0.8, 0, witness)["config"]["witness"] == witness
    assert record.payload({}, 0.8, 0)["config"]["witness"] == {}, "a run without a holdout has nothing to witness"


def test_payload_records_the_runs_own_crop():
    """The member stamps the crop the RUN used, not the module constant.

    Stamping the constant would record 130 mm about the image centre for an arm trained at 80 mm about
    the tissue — and the serve path reads this field, so the member would then be fed pixels it was
    never fitted on, with every fingerprint still passing against its own reading.
    """
    record = replace(_record(), options=replace(OPTIONS, crop_mm=80.0, centre="tissue"))

    config = record.payload({}, holdout=0.8, epoch=1)["config"]

    assert config["crop_mm"] == 80.0
    assert config["centre"] == "tissue"
    assert _record().payload({}, holdout=0.8, epoch=1)["config"]["crop_mm"] == CROP_MM


def test_payload_records_the_runs_own_router():
    """The member stamps the ROUTER the run used, so serving cannot fall back to ours.

    A graft-matching arm fitted under last-maximal and served under first would be fed a different
    series in every contested slot - fitted on one set of images and shown another, with the fingerprint
    still passing because it is checked against the member's own reading.
    """
    record = replace(_record(), options=replace(OPTIONS, tie="last"))

    assert record.payload({}, holdout=0.8, epoch=1)["config"]["tie"] == "last"
    assert _record().payload({}, holdout=0.8, epoch=1)["config"]["tie"] == "first"


def test_of(tmp_path: Path):
    """The stamp read back from a checkpoint, round-tripped through the assembler that writes it.

    Round-tripped rather than hand-written, because a hand-written payload only proves the copy agrees
    with itself — and what this reader exists for is the OLDER half of the ledger, whose only account
    of its own split is this block.
    """
    record = _record()
    checkpoint = tmp_path / "arm.pt"
    torch.save(record.payload({"w": torch.zeros(1)}, holdout=0.83, epoch=6), checkpoint)

    stamped = StampedRun.of(checkpoint)

    assert stamped is not None
    assert stamped.options["whole_stack"] == "True", "the run's flags, as the recipe stores them"
    assert stamped.kept_epoch == 6


def test_of_reads_the_kept_epoch_from_a_checkpoint_that_predates_the_epoch_key(tmp_path: Path):
    """`epochs_done` is the kept epoch plus one, and for most of the ledger it is the only one written."""
    record = _record()
    payload = record.payload({"w": torch.zeros(1)}, holdout=0.83, epoch=6)
    del payload["epoch"]
    checkpoint = tmp_path / "older.pt"
    torch.save(payload, checkpoint)

    stamped = StampedRun.of(checkpoint)

    assert stamped is not None
    assert stamped.kept_epoch == 6, "not epochs_done, which counts the run rather than naming the save"


def test_of_returns_nothing_for_a_checkpoint_with_no_stamp(tmp_path: Path):
    """14 members predate options entirely: name and score is all they have, and that must be said."""
    checkpoint = tmp_path / "ancient.pt"
    torch.save({"model": {"w": torch.zeros(1)}, "holdout": 0.81}, checkpoint)

    assert StampedRun.of(checkpoint) is None
    assert StampedRun.of(tmp_path / "absent.pt") is None
