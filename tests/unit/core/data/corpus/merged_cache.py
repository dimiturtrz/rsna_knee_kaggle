"""Two corpora concatenated into the one cache the trainer reads."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data.corpus.merged_cache import MergedCache
from core.data.corpus.slots import MERGED_SCHEME, RIJEKA_SCHEME
from core.data.corpus.study_cache import StudyCache

SLICES = 2
SIZE = 2


def written(directory: Path, scheme: str | None, studies: list[str], fill: int) -> StudyCache:
    """A finished cache on disk: images, index and mask, at a size a unit test can hold."""
    cache = StudyCache(directory, slices=SLICES, image_size=SIZE)
    cache = cache if scheme is None else cache.under(scheme)
    cache.prepare()
    np.save(cache.images_path, np.full(cache.shape(len(studies)), fill, dtype=np.uint8))
    cache.record(studies)
    np.save(cache.mask_path, np.ones((len(studies), cache.slots), dtype=np.float32))
    return cache


def merger(directory: Path, ours: list[str], theirs: list[str]) -> MergedCache:
    """Our corpus, a one-slot foreign one, and the merged destination they write into."""
    base = written(directory, None, ours, fill=1)
    return MergedCache(base, written(directory, RIJEKA_SCHEME, theirs, fill=2), base.under(MERGED_SCHEME))


def test_positions(tmp_path: Path):
    """A foreign corpus's single slot lands at the canonical slot of the same NAME, not at index 0 by luck."""
    merged = merger(tmp_path, ["a"], ["b"])

    assert merged.positions() == [merged.merged.scheme_slots.index(merged.extra.scheme_slots[0])]


def test_check(tmp_path: Path):
    """A destination that is one of its own sources is refused before anything is written."""
    base = written(tmp_path, None, ["a"], fill=1)

    with pytest.raises(ValueError, match="destination is a source"):
        MergedCache(base, base.under(RIJEKA_SCHEME), base).check()


def test_build(tmp_path: Path):
    """Ours first at every slot, theirs after at their matched slot only, with the mask saying so."""
    merged = merger(tmp_path, ["a", "b"], ["x"])

    merged.build()

    images = np.load(merged.merged.images_path)
    mask = np.load(merged.merged.mask_path)
    slot = merged.positions()[0]
    assert json.loads(merged.merged.index_path.read_text(encoding="utf-8")) == ["a", "b", "x"]
    assert (images[:2] == 1).all()
    assert (images[2, slot] == 2).all()
    assert mask[2].tolist() == [1.0 if index == slot else 0.0 for index in range(merged.merged.slots)]


def test_build_refuses_a_shared_identifier(tmp_path: Path):
    """A study in both corpora would be weighted twice, so the merge stops rather than duplicating it."""
    merged = merger(tmp_path, ["a"], ["a"])

    with pytest.raises(ValueError, match="in both corpora"):
        merged.build()


def test_labels(tmp_path: Path):
    """The union of columns: each study scored on what ITS table named, NaN — weight 0 — on the rest."""
    merged = merger(tmp_path, ["a"], ["x"])
    ours, theirs = tmp_path / "ours.csv", tmp_path / "theirs.csv"
    pd.DataFrame({"StudyInstanceUID": ["a"], "ACL": [0.9], "Effusion": [0.2]}).to_csv(ours, index=False)
    pd.DataFrame({"StudyInstanceUID": ["x"], "ACL": [0.82]}).to_csv(theirs, index=False)

    merged.labels(ours, theirs, tmp_path / "both.csv")

    table = pd.read_csv(tmp_path / "both.csv").set_index("StudyInstanceUID")
    assert table.loc["x", "ACL"] == 0.82
    assert table["Effusion"].isna().tolist() == [False, True]


def test_labels_carry_corpus_tags(tmp_path: Path):
    """The merged table names each row's corpus so Scheme B sampling does not guess from identifiers."""
    merged = merger(tmp_path, ["a"], ["x"])
    ours, theirs = tmp_path / "ours.csv", tmp_path / "theirs.csv"
    pd.DataFrame({"StudyInstanceUID": ["a"], "ACL": [0.9]}).to_csv(ours, index=False)
    pd.DataFrame({"StudyInstanceUID": ["x"], "ACL": [0.82]}).to_csv(theirs, index=False)

    merged.labels(ours, theirs, tmp_path / "both.csv", base_name="canonical", extra_name="rijeka")

    table = pd.read_csv(tmp_path / "both.csv")
    assert "corpus" in table.columns
    assert table["corpus"].tolist() == ["canonical", "rijeka"]
