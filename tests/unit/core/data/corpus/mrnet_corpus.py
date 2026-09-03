import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.data.corpus.mrnet_corpus import (
    ACL,
    CONFIDENCE,
    MRNET_CORPUS,
    MRNET_LEVEL,
    MRNET_MATRIX,
    MRNET_SPACING,
    PLANES,
    PREFIX,
    REPORT_CONFIDENCE,
    MRNetCorpus,
    MRNetExam,
)
from core.data.corpus.mrnet_labels import TableLabels
from core.data.corpus.slots import MRNET_SCHEME, SLOTS, Routing
from core.data.corpus.study_cache import StudyCache
from core.findings import STUDY

DEPTH, MATRIX = 6, 32
ROOT = "MRNet-v1.0"
TRAIN_CASES = ("0000", "0001", "0002")
VALID_CASE = "1130"


def _volume() -> bytes:
    """One plane's stack in the archive's own format — a plain `.npy`, never a pickle."""
    stack = np.arange(DEPTH * MATRIX * MATRIX, dtype=np.uint8).reshape(DEPTH, MATRIX, MATRIX)
    buffer = io.BytesIO()
    np.save(buffer, stack)
    return buffer.getvalue()


def _archive(tmp_path: Path, shipped: int = 2) -> Path:
    """A miniature MRNet archive: three labelled train cases, `shipped` of whose volumes are here.

    The gap is the point — the paper names 1,370 exams and 1,250 download, and the join is what keeps
    every rate taken from it over the exams whose pixels are present.
    """
    path = tmp_path / "MRNet-v1.0.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{ROOT}/train-acl.csv", "".join(f"{case},{index % 2}\n" for index, case in enumerate(TRAIN_CASES))
        )
        archive.writestr(f"{ROOT}/valid-acl.csv", f"{VALID_CASE},1\n")
        for case in TRAIN_CASES[:shipped]:
            for plane in PLANES:
                archive.writestr(f"{ROOT}/train/{plane}/{case}.npy", _volume())
        for plane in PLANES:
            archive.writestr(f"{ROOT}/valid/{plane}/{VALID_CASE}.npy", _volume())
    return path


def _flat_archive(tmp_path: Path) -> Path:
    """The Redivis export's layout: `<plane>/<case>.npy` and NO label table anywhere in the zip.

    Both differences are measured, not assumed — the 7.74 GB download of 2026-08-29 held 3,750 arrays,
    1,250 per plane, case 0000 to 1249, and zero non-`.npy` members.
    """
    path = tmp_path / "flat.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for case in (*TRAIN_CASES[:2], VALID_CASE):
            for plane in PLANES:
                archive.writestr(f"{plane}/{case}.npy", _volume())
    return path


def _tables(tmp_path: Path) -> Path:
    """The ACL tables as a separate download, unpadded and with a header — the two ways an export differs."""
    directory = tmp_path / "tables"
    directory.mkdir()
    (directory / "mrnet-train-acl.csv").write_text("case,acl\n0,0\n1,1\n", encoding="utf-8")
    (directory / "mrnet-valid-acl.csv").write_text(f"case,acl\n{int(VALID_CASE)},1\n", encoding="utf-8")
    return directory


def _cache(tmp_path: Path) -> StudyCache:
    """A tiny cache under the scheme that keeps this corpus off our own arrays."""
    return StudyCache(tmp_path, slices=4, image_size=8, routing=Routing(scheme=MRNET_SCHEME))


def test_level():
    """A negative and a positive row land on the levels whose MEASURED agreement they match."""
    assert MRNetExam("a", {}, 0).level() == MRNET_LEVEL[0]
    assert MRNetExam("a", {}, 1).level() == MRNET_LEVEL[1]
    assert MRNET_LEVEL[0] < MRNET_LEVEL[1], "the ordering is what carries the mechanism"


def test_present():
    """The mask says which planes an exam actually ships, so no black slot is called acquired."""
    exam = MRNetExam("a", {PLANES[0]: "x.npy", PLANES[2]: "z.npy"}, 1)

    assert exam.present(3) == [1.0, 0.0, 1.0]


def test_m_r_net_corpus_volumes(tmp_path: Path):
    """Only exams whose PIXELS ship are returned, across both downloadable splits and prefixed.

    Joining on the label tables' length instead would report rates over exams that are not here, which
    is the 1,370-vs-1,250 error this corpus invites.
    """
    corpus = MRNetCorpus(_archive(tmp_path), _cache(tmp_path))

    exams = corpus.volumes()

    assert [exam.identifier for exam in exams] == [f"{PREFIX}0000", f"{PREFIX}0001", f"{PREFIX}{VALID_CASE}"]
    assert len(exams) == 3, "the third train case is labelled but its volumes are absent"
    assert sorted(exams[0].members) == sorted(PLANES), "a complete exam ships all three planes"
    assert exams[1].acl == 1


def test_m_r_net_corpus_volumes_honours_the_cache_truncation(tmp_path: Path):
    """`--cache-studies` is in the cache's NAME, so the exam list has to be cut to match it."""
    cache = StudyCache(tmp_path, slices=4, image_size=8, routing=Routing(scheme=MRNET_SCHEME), first=2)

    exams = MRNetCorpus(_archive(tmp_path), cache).volumes()

    assert [exam.identifier for exam in exams] == [f"{PREFIX}0000", f"{PREFIX}0001"]


def test_m_r_net_corpus_volumes_reads_a_flat_archive_with_external_tables(tmp_path: Path):
    """Pixels with no split directory and labels from outside the zip build the same exam list.

    This is the export we actually hold: `<plane>/<case>.npy` and not one CSV. If the split could not be
    recovered from the case number, the 120 validation exams would be unreachable.
    """
    corpus = MRNetCorpus(_flat_archive(tmp_path), _cache(tmp_path), tables=TableLabels(_tables(tmp_path)))

    exams = corpus.volumes()

    assert [exam.identifier for exam in exams] == [f"{PREFIX}0000", f"{PREFIX}0001", f"{PREFIX}{VALID_CASE}"]
    assert sorted(exams[2].members) == sorted(PLANES), "the validation exam is found without a valid/ directory"
    assert exams[2].acl == 1


def test_m_r_net_corpus_metadata(tmp_path: Path):
    """Every shipped exam gets a row naming its corpus, the planes it ships, and its geometry."""
    rows = MRNetCorpus(_archive(tmp_path), _cache(tmp_path)).metadata()

    assert [row.identifier for row in rows] == [f"{PREFIX}0000", f"{PREFIX}0001", f"{PREFIX}{VALID_CASE}"]
    assert {row.corpus for row in rows} == {MRNET_CORPUS}
    assert rows[0].plane == "Sagittal+Axial+Coronal", "one flat cell naming the SET this exam ships"
    assert rows[0].pixel_spacing_mm == MRNET_SPACING
    assert rows[0].matrix_px == MRNET_MATRIX


def test_m_r_net_corpus_metadata_leaves_the_sequence_absent(tmp_path: Path):
    """MRNet publishes no per-exam sequence, so the three sequence axes are unanswerable and stay None.

    Deriving them from the plane would put a guess in a cell that reads identically to Rijeka's, whose
    same cells come from a published protocol.
    """
    row = MRNetCorpus(_archive(tmp_path), _cache(tmp_path)).metadata()[0]

    assert row.sequence is None
    assert row.fat_suppressed is None
    assert row.fluid_sensitive is None
    assert row.age_years is None


def test_stack(tmp_path: Path):
    """A plane becomes the cache's own shape, band-sampled and percentile-scaled by the shared reader."""
    corpus = MRNetCorpus(_archive(tmp_path), _cache(tmp_path))
    volume = np.arange(DEPTH * MATRIX * MATRIX, dtype=np.float32).reshape(DEPTH, MATRIX, MATRIX)

    stacked = corpus.stack(volume, slices=4)

    assert stacked.shape == (4, 8, 8), "slices asked for, at the cache's in-plane size"
    assert stacked.dtype == torch.uint8, "the cache holds eight bits per pixel and so must this"
    assert int(stacked.max()) - int(stacked.min()) > 250, "scaled to its own 1st-99th percentiles"


def test_stack_crops_when_the_physical_extent_fits(tmp_path: Path):
    """The physical crop is taken when the spacing admits one, and skipped when it would exceed the frame.

    Both branches matter, and this corpus sits near the boundary: 130 mm is 208 px of 256 at 0.625 mm/px
    and 260 px at 0.5, so a wrong spacing here silently declines to crop and hands the encoder a
    different mm/pixel than every member was fitted at.
    """
    volume = np.arange(DEPTH * MATRIX * MATRIX, dtype=np.float32).reshape(DEPTH, MATRIX, MATRIX)
    coarse = MRNetCorpus(_archive(tmp_path), _cache(tmp_path), spacing=130.0 / 20.0)
    fine = MRNetCorpus(_archive(tmp_path), _cache(tmp_path), spacing=0.625)

    assert coarse.stack(volume, slices=4).shape == (4, 8, 8)
    assert not np.array_equal(coarse.stack(volume, slices=4).numpy(), fine.stack(volume, slices=4).numpy()), (
        "a 130 mm crop at 6.5 mm/px takes 20 of 32 pixels; at 0.625 mm/px it wants 208 and cannot"
    )


def test_build(tmp_path: Path):
    """The three cache files land, shaped as the trainer loads them, with the mask written LAST."""
    cache = _cache(tmp_path)
    corpus = MRNetCorpus(_archive(tmp_path), cache)

    corpus.build()

    corpus_images, identifiers = cache.load()
    assert corpus_images.images.shape == (3, 3, 4, 8, 8), "three slots, because the corpus acquires three planes"
    assert identifiers == [f"{PREFIX}0000", f"{PREFIX}0001", f"{PREFIX}{VALID_CASE}"]
    assert corpus_images.mask.shape == (3, 3) and float(corpus_images.mask.min()) == 1.0
    assert int(corpus_images.images[0, 1].max()) > 0, "every plane is decoded, not only the first slot"
    assert json.loads(cache.index_path.read_text(encoding="utf-8")) == identifiers


def test_build_writes_under_a_key_that_cannot_hit_our_own_corpus(tmp_path: Path):
    """The scheme suffixes the cache name, which is the whole reason a foreign corpus gets a scheme.

    Under the canonical scheme this build would write `images_336px_12sl.npy` — the arrays every member
    on disk was fitted on — and destroy them with something differing only in its first axis.
    """
    cache = _cache(tmp_path)
    MRNetCorpus(_archive(tmp_path), cache).build()

    assert cache.key.endswith(f"_{MRNET_SCHEME}")
    assert cache.images_path.name == f"images_{cache.key}.npy"
    assert not (tmp_path / "images_8px_4sl.npy").exists(), "never the unsuffixed name"
    assert cache.slots == 3 < len(SLOTS)


def test_m_r_net_corpus_labels(tmp_path: Path):
    """The table carries ACL and its confidence and NO column for the other eleven findings.

    Absence is the mechanism: `Targets.pair` gives a missing column weight 0.0, so eleven heads take no
    gradient from this corpus. The meniscus label is dropped rather than written into both meniscal
    columns, which would teach a medial tear as lateral-positive.
    """
    corpus = MRNetCorpus(_archive(tmp_path), _cache(tmp_path))
    destination = tmp_path / "labels" / "mrnet_acl.csv"

    corpus.labels(destination)

    table = pd.read_csv(destination)
    assert list(table.columns) == [STUDY, ACL, CONFIDENCE]
    assert table[STUDY].tolist() == [f"{PREFIX}0000", f"{PREFIX}0001", f"{PREFIX}{VALID_CASE}"]
    assert table[ACL].tolist() == [MRNET_LEVEL[0], MRNET_LEVEL[1], MRNET_LEVEL[1]]
    assert set(table[CONFIDENCE]) == {REPORT_CONFIDENCE}


def test_m_r_net_corpus_write_labels(tmp_path: Path):
    """The shared writer emits the three columns both producers owe, and creates the directory."""
    destination = tmp_path / "labels" / "mrnet_acl.csv"

    MRNetCorpus.write_labels([("mrnet-0000", MRNET_LEVEL[0]), ("mrnet-0001", MRNET_LEVEL[1])], destination)

    table = pd.read_csv(destination)
    assert list(table.columns) == [STUDY, ACL, CONFIDENCE]
    assert table[ACL].tolist() == [MRNET_LEVEL[0], MRNET_LEVEL[1]]
    assert set(table[CONFIDENCE]) == {REPORT_CONFIDENCE}
