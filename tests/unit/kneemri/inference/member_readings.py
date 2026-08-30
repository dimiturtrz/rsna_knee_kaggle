"""Members grouped by the cache each one names, on tiny caches built here so the grouping is known."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from core.data.corpus.study_cache import StudyCache
from kneemri.inference.member_readings import MemberReadings
from kneemri.inference.members import Member, MemberConfig
from tests.unit.conftest import write_series


def _member(directory: Path, name: str, *, crop_mm: float = 130.0) -> Member:
    """A member recording one reading, whose checkpoint exists because the prediction key stats it."""
    checkpoint = directory / f"{name}.pt"
    checkpoint.write_bytes(b"weights")
    return Member(
        identifier=name,
        file=checkpoint,
        fold=0,
        config={
            "img": 8,
            "slices": 2,
            "group": 1,
            "crop_mm": crop_mm,
            "band": [0.2, 0.8],
            "slots": ["a"],
            "variant": "small",
            "pool": "cls_mean",
        },
        fingerprint=np.zeros((1, 12), np.float32),
        holdout=0.8,
        annot=0.8,
    )


def _cache(directory: Path, source: Path, *, crop_mm: float = 130.0) -> StudyCache:
    """A one-study cache under exactly the reading a member of that crop names."""
    cache = _member(directory, "probe", crop_mm=crop_mm).cache(directory)
    cache.build(source, ["study0"], {"sag": "Sagittal"}, workers=1)
    return cache


def test_keys(tmp_path: Path):
    """Members that read the same pixels share ONE cache; a different crop is a different cache.

    THE DEFECT THIS PINS: grouping used to be by size, depth and scheme — three of the six fields that
    name a cache — so two members differing only in crop or band were decoded as if they read the same
    array, and the ruler check did not group at all.
    """
    readings = MemberReadings(
        [_member(tmp_path, "a"), _member(tmp_path, "b"), _member(tmp_path, "c", crop_mm=80.0)], tmp_path
    )

    assert readings.keys == ["8px_2sl", "8px_2sl_80mm"], "first-listed first, one entry per distinct reading"
    assert [member.identifier for member in readings.owners["8px_2sl"]] == ["a", "b"]


def test_an_encoder_beside_the_members_is_skipped_rather_than_raising(tmp_path: Path):
    """A pool sweep globs the processed directory and gets the SSL encoders with it.

    They record pretraining knobs and no read size, so there is no cache to group them by. Grouping used
    to ask every checkpoint for its `image_size`, which raised `KeyError: 'img'` on the first encoder —
    before a single real member had been read, over a directory where 7 of 200 files are encoders.
    """
    encoder = replace(_member(tmp_path, "enc"), config=cast(MemberConfig, {"mask_ratio": 0.75}))

    readings = MemberReadings([_member(tmp_path, "a"), encoder], tmp_path)

    assert [member.identifier for member in readings.members] == ["a"]
    assert [member.identifier for member in readings.skipped] == ["enc"]
    assert readings.keys == ["8px_2sl"], "the encoder names no reading, so it adds no cache"


def test_reference(tmp_path: Path):
    """The first member's own cache is what the study order is taken from."""
    source = tmp_path / "src"
    write_series(source / "study0" / "sag", slices=4, description="SAG PD FS", size=32)
    _cache(tmp_path, source)

    corpus, identifiers = MemberReadings([_member(tmp_path, "a")], tmp_path).reference()

    assert identifiers == ["study0"]
    assert corpus.slices == 2, "the corpus reports the depth of the cache the member named"


def test_slabs(tmp_path: Path):
    """Each member arrives with the holdout of its OWN cache, and a disagreeing order is refused."""
    source = tmp_path / "src"
    write_series(source / "study0" / "sag", slices=4, description="SAG PD FS", size=32)
    _cache(tmp_path, source)
    _cache(tmp_path, source, crop_mm=80.0)
    readings = MemberReadings([_member(tmp_path, "a"), _member(tmp_path, "b", crop_mm=80.0)], tmp_path)
    corpus, identifiers = readings.reference()
    holdout = np.array([0])

    grouped = list(readings.slabs(holdout, identifiers, corpus))

    assert [member.identifier for member, _, _ in grouped] == ["a", "b"]
    assert grouped[0][1] is not grouped[1][1], "a different cache is a different slab, not the reference one"
    assert all(held.slices == 2 for _, _, held in grouped)

    with pytest.raises(ValueError, match="different order"):
        list(readings.slabs(holdout, ["someone else"], corpus))
