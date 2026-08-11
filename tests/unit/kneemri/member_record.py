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
    slices: int = 12
    slice_pooling: str = "max"
    anatomical_prior: bool = False
    ema: bool = False
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
