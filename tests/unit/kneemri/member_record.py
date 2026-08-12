from dataclasses import dataclass
from pathlib import Path

import torch

from kneemri.backbone import Encoder
from kneemri.member_record import MemberRecord

from .model import SLOTS, _FakeBackbone  # the stand-in encoder this suite already trusts

DIM = 8


@dataclass
class _Options:
    """The fields MemberRecord reads. A stand-in so the test does not need a parsed command line."""

    image_size: int = 336
    train_size: int = 0
    slices: int = 12
    slice_pooling: str = "max"
    anatomical_prior: bool = False
    ema: bool = False
    bf16: bool = False
    lora_rank: int = 0
    rank_weight: float = 1.0
    report_weight: float = 0.0
    whole_stack: bool = True
    unfreeze_last: int = 6
    head_lr: float = 1e-3
    backbone_lr: float = 4e-5
    out: Path | None = None


def _record() -> MemberRecord:
    from kneemri.model import StudyModel  # noqa: PLC0415 - keeps the fake backbone import local

    torch.manual_seed(2026)
    model = StudyModel(_FakeBackbone(), DIM, SLOTS)
    encoder = Encoder(variant="small", repository="facebook/dinov2-small", dim=DIM)
    return MemberRecord(_Options(), encoder, model, torch.device("cpu"), [f"S{i}" for i in range(SLOTS)])


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

    resampled = replace(record, options=_Options(train_size=392))
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
