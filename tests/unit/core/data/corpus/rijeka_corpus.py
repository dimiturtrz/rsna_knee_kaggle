import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.data.corpus.rijeka_corpus import ACL, CONFIDENCE, READER_LEVEL, RijekaCorpus, RijekaVolume
from core.data.corpus.slots import RIJEKA_SCHEME, SLOTS, Routing
from core.data.corpus.study_cache import StudyCache
from core.findings import STUDY

DEPTH, MATRIX = 6, 32
PUBLISHED = 3


def _archive(tmp_path: Path, present: int = 2) -> Path:
    """A miniature kneeMRI archive: three published exams, `present` of whose volumes actually ship.

    The gap is the point — the real mirror lists 917 rows and holds 736 volumes, and the join is what
    keeps every rate taken from it over the exams whose pixels are here.
    """
    rows = [
        {"examId": 100 + index, "seriesNo": 8, "aclDiagnosis": index % len(READER_LEVEL)}
        | {"volumeFilename": f"{100 + index}-8.pck"}
        for index in range(PUBLISHED)
    ]
    path = tmp_path / "kneemri.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("kneemri/metadata.csv", pd.DataFrame(rows).to_csv(index=False))
        for row in rows[:present]:
            volume = np.arange(DEPTH * MATRIX * MATRIX, dtype=np.uint16).reshape(DEPTH, MATRIX, MATRIX)
            archive.writestr(f"kneemri/vol01/{row['volumeFilename']}", pickle.dumps(volume))
    return path


def _cache(tmp_path: Path) -> StudyCache:
    """A tiny cache under the scheme that keeps this corpus off our own arrays."""
    return StudyCache(tmp_path, slices=4, image_size=8, routing=Routing(scheme=RIJEKA_SCHEME))


def test_level():
    """Each of the three ligament verdicts lands on the level whose MEASURED agreement it matches."""
    assert RijekaVolume("a", "m", 0).level() == READER_LEVEL[0]
    assert RijekaVolume("a", "m", 2).level() == READER_LEVEL[2]
    assert READER_LEVEL[0] < READER_LEVEL[1] < READER_LEVEL[2], "the ordering is what carries the mechanism"


def test_volumes(tmp_path: Path):
    """Only exams whose PIXELS ship are returned — the published table is longer than the archive.

    Joining on the table's length instead would report rates over volumes that are not there, which is
    the 917-vs-736 error this corpus invites.
    """
    corpus = RijekaCorpus(_archive(tmp_path), _cache(tmp_path))

    volumes = corpus.volumes()

    assert len(volumes) == 2, "the third exam is published but absent from this archive"
    assert [volume.identifier for volume in volumes] == ["100-8", "101-8"]
    assert volumes[0].member.endswith("100-8.pck")
    assert volumes[1].diagnosis == 1


def test_stack(tmp_path: Path):
    """A volume becomes the cache's own shape, band-sampled and percentile-scaled by the shared reader."""
    corpus = RijekaCorpus(_archive(tmp_path), _cache(tmp_path))
    volume = np.arange(DEPTH * MATRIX * MATRIX, dtype=np.float32).reshape(DEPTH, MATRIX, MATRIX)

    stacked = corpus.stack(volume, slices=4)

    assert stacked.shape == (4, 8, 8), "slices asked for, at the cache's in-plane size"
    assert stacked.dtype == torch.uint8, "the cache holds eight bits per pixel and so must this"
    assert int(stacked.max()) - int(stacked.min()) > 250, "scaled to its own 1st-99th percentiles"


def test_stack_crops_when_the_physical_extent_fits(tmp_path: Path):
    """The physical crop is taken when the spacing admits one, and skipped when it would exceed the frame.

    Both branches matter: the real corpus at 0.5 mm/px crops 260 px of 320, and a wrong spacing that
    silently declined to crop would hand the encoder a different mm/pixel than every member was fitted at.
    """
    volume = np.arange(DEPTH * MATRIX * MATRIX, dtype=np.float32).reshape(DEPTH, MATRIX, MATRIX)
    coarse = RijekaCorpus(_archive(tmp_path), _cache(tmp_path), spacing=130.0 / 20.0)
    fine = RijekaCorpus(_archive(tmp_path), _cache(tmp_path), spacing=0.5)

    assert coarse.stack(volume, slices=4).shape == (4, 8, 8)
    assert not np.array_equal(coarse.stack(volume, slices=4).numpy(), fine.stack(volume, slices=4).numpy()), (
        "a 130 mm crop at 6.5 mm/px takes 20 of 32 pixels; at 0.5 mm/px it wants 260 and cannot"
    )


def test_build(tmp_path: Path):
    """The three cache files land, shaped as the trainer loads them, with the mask written LAST."""
    cache = _cache(tmp_path)
    corpus = RijekaCorpus(_archive(tmp_path), cache)

    corpus.build()

    images, identifiers = cache.load()
    assert images.images.shape == (2, 1, 4, 8, 8), "one slot, because the corpus acquired one sequence"
    assert identifiers == ["100-8", "101-8"]
    assert images.mask.shape == (2, 1) and float(images.mask.min()) == 1.0, "no slot for the mask to deny"
    assert json.loads(cache.index_path.read_text(encoding="utf-8")) == identifiers


def test_build_writes_under_a_key_that_cannot_hit_our_own_corpus(tmp_path: Path):
    """The scheme suffixes the cache name, which is the whole reason a foreign corpus gets a scheme.

    Under the canonical scheme this build would write `images_336px_12sl.npy` — the arrays every member
    on disk was fitted on — and destroy them with something differing only in its first axis.
    """
    cache = _cache(tmp_path)
    RijekaCorpus(_archive(tmp_path), cache).build()

    assert cache.key.endswith(f"_{RIJEKA_SCHEME}")
    assert cache.images_path.name == f"images_{cache.key}.npy"
    assert not (tmp_path / "images_8px_4sl.npy").exists(), "never the unsuffixed name"
    assert cache.slots == 1 < len(SLOTS)


def test_labels(tmp_path: Path):
    """The table carries ACL and its confidence and NO column for the other eleven findings.

    Absence is the mechanism: `Targets.pair` gives a missing column weight 0.0, so eleven heads take no
    gradient from this corpus. Writing 0.5 into them would instead teach that every knee here is average
    on eleven findings nobody looked at.
    """
    corpus = RijekaCorpus(_archive(tmp_path), _cache(tmp_path))
    destination = tmp_path / "labels" / "rijeka_acl.csv"

    corpus.labels(destination)

    table = pd.read_csv(destination)
    assert list(table.columns) == [STUDY, ACL, CONFIDENCE]
    assert table[STUDY].tolist() == ["100-8", "101-8"]
    assert table[ACL].tolist() == [READER_LEVEL[0], READER_LEVEL[1]]
    assert set(table[CONFIDENCE]) == {1.0}
