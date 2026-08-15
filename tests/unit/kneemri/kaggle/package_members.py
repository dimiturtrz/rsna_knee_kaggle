"""The manifest rows, which are the contract between a checkpoint on disk and every kernel.

Nothing here needs a data root or a checkpoint file: a row is a pure function of a path and a payload,
and that is the point — while it lived inside `main` it could only be exercised by packaging 8 GB.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from kneemri.kaggle.package_members import FORMAT, MemberManifest, PackagedMember

CONFIG = {"crop_mm": 140.0, "whole_stack": True}
FINGERPRINT = [0.1, 0.2]


def _member(name: str, holdout: float = 0.84) -> PackagedMember:
    return PackagedMember(Path(name), {"config": CONFIG, "fingerprint": FINGERPRINT, "holdout": holdout})


def test_identifier():
    """The id comes off the filename by string surgery, and a member is addressed by it."""
    assert _member("member_fold3.pt").identifier == "fold3"
    assert _member("member_seed7_fold0.pt").identifier == "seed7_fold0"


def test_entry():
    """The row, at the keys `Member` reads back — fold parsed out of the id."""
    entry = _member("member_fold3.pt").entry()

    assert entry["id"] == "fold3"
    assert entry["fold"] == 3
    assert entry["file"] == "member_fold3.pt"
    assert entry["config"] == CONFIG
    assert entry["holdout"] == pytest.approx(0.84)
    assert np.isnan(float(str(entry["annot"]))), "the annotation check is the competition's, not ours"


def test_a_member_whose_name_carries_no_fold_packages_as_fold_zero():
    """`member_res392.pt` and `member_seed7_fold0.pt` are both real names in the processed root."""
    assert _member("member_res392.pt").entry()["fold"] == 0
    assert _member("member_seed7_fold0.pt").entry()["fold"] == 0


def test_a_full_data_fit_packages_with_no_holdout_rather_than_a_stand_in():
    """A fit with nothing held back records NaN by construction, and that must survive the manifest.

    Rewriting it as a number here would undo `fitting.NO_HOLDOUT`: a final fit advertising a score it
    never earned outranks every real member (knee-2d7).
    """
    member = _member("member_final.pt", holdout=float("nan"))

    assert np.isnan(float(str(member.entry()["holdout"])))
    assert np.isnan(json.loads(MemberManifest([member]).document())["members"][0]["holdout"])


def test_document():
    """What a mounted package is read from, at the format version `MemberPackage` expects."""
    document = json.loads(MemberManifest([_member("member_fold0.pt")]).document())

    assert document["format"] == FORMAT
    assert [member["id"] for member in document["members"]] == ["fold0"]


def test_two_checkpoints_sharing_one_id_are_refused_rather_than_overwriting_each_other():
    """Several globs can match the same NAME under different directories — the copy keeps one file.

    The manifest would then list two members and address one set of weights twice, which loads and
    fingerprints cleanly and predicts from something one of those rows never saw.
    """
    with pytest.raises(ValueError, match="one id"):
        MemberManifest([_member(f"{parent}/member_fold0.pt") for parent in ("a", "b")]).document()
