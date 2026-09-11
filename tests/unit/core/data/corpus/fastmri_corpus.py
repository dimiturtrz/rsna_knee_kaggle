import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from core.data.corpus.fastmri_corpus import (
    ACQUISITION,
    CARTILAGE_LOSS,
    FASTMRI_CORPUS,
    FASTMRI_PLANE,
    FASTMRI_SEQUENCE,
    FAT_SUPPRESSED_ACQUISITION,
    HEADER,
    NEGATIVE_LEVEL,
    PLAIN_ACQUISITION,
    POSITIVE_LEVEL,
    READER_CONFIDENCE,
    READINGS,
    RECONSTRUCTION,
    FastmriCorpus,
    Reading,
)
from core.data.corpus.slots import FASTMRI_SCHEME, SLOTS, Routing
from core.data.corpus.study_cache import StudyCache
from core.findings import STUDY

DEPTH, MATRIX = 6, 32
RECON_MATRIX, RECON_FOV = 320, 140.0
SPACING = RECON_FOV / RECON_MATRIX

HEADER_XML = (
    "<ismrmrdHeader><encoding><encodedSpace>"
    "<matrixSize><x>640</x><y>372</y></matrixSize>"
    "<fieldOfView_mm><x>280.0</x><y>280.0</y></fieldOfView_mm>"
    "</encodedSpace><reconSpace>"
    f"<matrixSize><x>{RECON_MATRIX}</x><y>{RECON_MATRIX}</y></matrixSize>"
    f"<fieldOfView_mm><x>{RECON_FOV}</x><y>{RECON_FOV}</y></fieldOfView_mm>"
    "</reconSpace></encoding></ismrmrdHeader>"
)
"""A header carrying BOTH spaces, in the order ISMRMRD writes them.

Both halves are the test: the encoded space comes first and would give 280/640 for pixels that are not
the ones being annotated, and inside each space `matrixSize` precedes `fieldOfView_mm`, which is the
ordering a pattern written the intuitive way round silently fails to match.
"""


def _volume(root: Path, name: str, acquisition: str) -> Path:
    """One exam in the shape the corpus reads it: recon volume, header, acquisition attribute."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.h5"
    pixels = np.arange(DEPTH * MATRIX * MATRIX, dtype=np.float32).reshape(DEPTH, MATRIX, MATRIX)
    with h5py.File(path, "w") as handle:
        handle.attrs[ACQUISITION] = acquisition
        handle.create_dataset(HEADER, data=np.bytes_(HEADER_XML.encode()))
        handle.create_dataset(RECONSTRUCTION, data=pixels * 1e-8)
    return path


def _root(tmp_path: Path) -> Path:
    """Three exams: one fat-suppressed, one not, one on an acquisition this scheme does not route."""
    root = tmp_path / "fastmri"
    _volume(root / "singlecoil_train", "file100", FAT_SUPPRESSED_ACQUISITION)
    _volume(root / "singlecoil_train", "file101", PLAIN_ACQUISITION)
    _volume(root / "singlecoil_train", "file102", "AXT2")
    return root


def _annotations(
    tmp_path: Path,
    listed: tuple[str, ...] = ("file100", "file101", "file102"),
    boxed: tuple[str, ...] = ("file100", "file101"),
) -> Path:
    """A miniature fastMRI+ pair: the box table, and the FILE LIST that is the real denominator.

    `boxed` is a subset of `listed` here as it is in the corpus — 974 annotated files inside 1,173 read
    ones. The two are separate knobs because the gap between them is what supplies the negatives.
    """
    path = tmp_path / "Annotations"
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        {"file": "file100", "slice": 12, "label": "Ligament - ACL High Grade Sprain", "x": 150.0, "width": 20.0},
        {"file": "file100", "slice": 14, "label": "Meniscus Tear", "x": 80.0, "width": 20.0},
        {"file": "file100", "slice": 15, "label": "Meniscus Tear", "x": 230.0, "width": 20.0},
        {"file": "file101", "slice": 9, "label": "Ligament - ACL Low Grade sprain", "x": 100.0, "width": 10.0},
    ]
    boxes = pd.DataFrame([row for row in rows if row["file"] in boxed])
    boxes.to_csv(path / "knee.csv", index=False)
    pd.DataFrame({0: list(listed)}).to_csv(path / "knee_file_list.csv", index=False, header=False)
    return path


def _cache(tmp_path: Path) -> StudyCache:
    """A tiny cache under the scheme that keeps this corpus off our own arrays."""
    return StudyCache(tmp_path, slices=4, image_size=8, routing=Routing(scheme=FASTMRI_SCHEME))


def _corpus(tmp_path: Path) -> FastmriCorpus:
    return FastmriCorpus(_root(tmp_path), _annotations(tmp_path), _cache(tmp_path))


def test_verdict():
    """A term at our threshold reads positive, a term that cannot be located reads UNKNOWN, silence reads negative.

    The middle case is the whole mapping: `Meniscus Tear` carries no side, so it must neither assert two
    lesions where one was read nor claim an absence the reader contradicted.
    """
    positive = Reading(positive=("Ligament - ACL High Grade Sprain",))
    coarse = Reading(coarse=("Meniscus Tear",))

    assert positive.verdict(frozenset({"Ligament - ACL High Grade Sprain"})) == POSITIVE_LEVEL
    assert coarse.verdict(frozenset({"Meniscus Tear"})) is None, "unlocatable blocks the negative, asserts nothing"
    assert positive.verdict(frozenset({"Meniscus Tear"})) == NEGATIVE_LEVEL
    assert positive.verdict(frozenset()) == NEGATIVE_LEVEL, "a read study with no box is an assessed absence"
    assert NEGATIVE_LEVEL < POSITIVE_LEVEL, "the ordering is what carries the mechanism"


def test_verdict_reads_a_graded_low_finding_as_a_negative():
    """`ACL Low Grade sprain` is not missing data: our criterion is high-grade, so it is a confident negative.

    Listing it as a third category would produce exactly the outcome silence produces, which is why
    `Reading` has two tuples and not three.
    """
    acl = Reading(positive=("Ligament - ACL High Grade Sprain",))

    assert acl.verdict(frozenset({"Ligament - ACL Low Grade sprain"})) == NEGATIVE_LEVEL


def test_volumes(tmp_path: Path):
    """Every exam whose acquisition this scheme routes, with its slot and its OWN spacing resolved.

    The third file is skipped rather than guessed into a slot — an unrouted acquisition would otherwise
    land in a channel meant for a different sequence, which no downstream check could see.
    """
    volumes = _corpus(tmp_path).volumes()

    assert [volume.identifier for volume in volumes] == ["file100", "file101"]
    assert [volume.slot for volume in volumes] == [0, 1], "fat-suppressed and plain fill DIFFERENT slots"


def test_spacing(tmp_path: Path):
    """In-plane spacing comes off the reconstructed field of view, not the acquisition matrix.

    The two differ on this corpus — an exam acquired on a coarser grid is reconstructed onto the finer
    one — and the physical crop downstream is sized in millimetres, so reading the wrong grid would
    silently crop the wrong amount of knee.
    """
    volumes = _corpus(tmp_path).volumes()

    assert volumes[0].spacing == SPACING, "read from reconSpace, not the encoded acquisition grid"


def test_metadata(tmp_path: Path):
    """Both routed exams get a row, and fat suppression follows the SLOT the acquisition string routed to.

    The unrouted third file has no row: `volumes` dropped it, so an acquisition nobody recognised never
    becomes a metadata cell.
    """
    rows = _corpus(tmp_path).metadata()

    assert [row.identifier for row in rows] == ["file100", "file101"]
    assert {row.corpus for row in rows} == {FASTMRI_CORPUS}
    assert (rows[0].plane, rows[0].sequence) == (FASTMRI_PLANE, FASTMRI_SEQUENCE)
    assert rows[0].fat_suppressed is True
    assert rows[1].fat_suppressed is False, "read back off the slot, not assumed for the corpus"
    assert rows[0].fluid_sensitive is True and rows[1].fluid_sensitive is True, "coronal PD either way"


def test_metadata_keeps_the_geometry_the_header_stated(tmp_path: Path):
    """Matrix and field of view are recorded, not only their quotient — 4 in 100 exams are acquired at 150 mm."""
    row = _corpus(tmp_path).metadata()[0]

    assert (row.matrix_px, row.field_of_view_mm) == (RECON_MATRIX, RECON_FOV)
    assert row.pixel_spacing_mm == SPACING
    assert row.field_strength_t is None, "the header carries it; this reader does not parse it"


def test_stack(tmp_path: Path):
    """A reconstruction becomes the cache's own shape, band-sampled and percentile-scaled by the shared reader."""
    corpus = _corpus(tmp_path)
    volume = np.arange(DEPTH * MATRIX * MATRIX, dtype=np.float32).reshape(DEPTH, MATRIX, MATRIX) * 1e-8

    stacked = corpus.stack(volume, SPACING, slices=4)

    assert stacked.shape == (4, 8, 8), "slices asked for, at the cache's in-plane size"
    assert stacked.dtype == torch.uint8, "the cache holds eight bits per pixel and so must this"
    assert int(stacked.max()) - int(stacked.min()) > 250, (
        "scaled to its own 1st-99th percentiles — these magnitudes peak near 1e-4 and a fixed window "
        "would return an empty image"
    )


def test_stack_crops_when_the_physical_extent_fits(tmp_path: Path):
    """The physical crop is taken when the spacing admits one and skipped when it would exceed the frame.

    Both branches matter: a wrong spacing that silently declined to crop would hand the encoder a
    different mm/pixel than every member on disk was fitted at.
    """
    corpus = _corpus(tmp_path)
    volume = np.arange(DEPTH * MATRIX * MATRIX, dtype=np.float32).reshape(DEPTH, MATRIX, MATRIX)

    fine = corpus.stack(volume, SPACING, slices=4)
    coarse = corpus.stack(volume, 130.0 / 20.0, slices=4)

    assert coarse.shape == fine.shape == (4, 8, 8)
    assert not np.array_equal(coarse.numpy(), fine.numpy()), (
        "a 130 mm crop at 6.5 mm/px takes 20 of 32 pixels; at 0.4375 mm/px it wants 297 and cannot"
    )


def test_build(tmp_path: Path):
    """The three cache files land, shaped as the trainer loads them, with the mask written LAST."""
    cache = _cache(tmp_path)
    corpus = FastmriCorpus(_root(tmp_path), _annotations(tmp_path), cache)

    corpus.build()

    images, identifiers = cache.load()
    assert images.images.shape == (2, 2, 4, 8, 8), "two exams, two slots"
    assert identifiers == ["file100", "file101"]
    assert json.loads(cache.index_path.read_text(encoding="utf-8")) == identifiers


def test_build_writes_a_mask_that_is_not_all_ones(tmp_path: Path):
    """Exactly one slot fills per exam, because a knee is acquired with fat suppression or without.

    This is the presence mask doing the job it exists for, and it is why this corpus can tell a foreign
    study apart from a degenerate one — a corpus that fills every slot it has cannot.
    """
    cache = _cache(tmp_path)
    FastmriCorpus(_root(tmp_path), _annotations(tmp_path), cache).build()

    images, _ = cache.load()
    assert images.mask.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_build_writes_under_a_key_that_cannot_hit_our_own_corpus(tmp_path: Path):
    """The scheme suffixes the cache name, which is the whole reason a foreign corpus gets a scheme.

    Under the canonical scheme this build would write `images_336px_12sl.npy` — the arrays every member
    on disk was fitted on — and destroy them with something differing only in its first axis.
    """
    cache = _cache(tmp_path)
    FastmriCorpus(_root(tmp_path), _annotations(tmp_path), cache).build()

    assert cache.key.endswith(f"_{FASTMRI_SCHEME}")
    assert cache.images_path.name == f"images_{cache.key}.npy"
    assert not (tmp_path / "images_8px_4sl.npy").exists(), "never the unsuffixed name"
    assert cache.slots == 2 < len(SLOTS)


def test_read(tmp_path: Path):
    """Every READ study appears, including the ones found clean — they are the negatives this corpus supplies.

    Keying on the box table alone would drop them and raise every rate reported off this corpus.
    """
    seen = FastmriCorpus(_root(tmp_path), _annotations(tmp_path), _cache(tmp_path)).read()

    assert set(seen) == {"file100", "file101", "file102"}
    assert seen["file100"] == frozenset({"Ligament - ACL High Grade Sprain", "Meniscus Tear"})
    assert seen["file102"] == frozenset(), "read and clean, not unread"


def test_read_keeps_the_first_study_out_of_the_column_name(tmp_path: Path):
    """`knee_file_list.csv` ships with NO header row, so a default read eats its first study.

    One row, and it is the difference between a rate and a rate that is wrong — every rate in this
    module's first draft was over 1,172 studies instead of 1,173 for exactly this reason.
    """
    seen = FastmriCorpus(_root(tmp_path), _annotations(tmp_path), _cache(tmp_path)).read()

    assert "file100" in seen, "the first listed study is a study, not a column name"
    assert len(seen) == 3


def test_labels(tmp_path: Path):
    """Eleven findings, most carrying absence; the unlocatable ones carry no column for that study at all.

    Absence of a column is the mechanism: `Targets.pair` gives it weight 0.0, so the head takes no
    gradient rather than learning a guess.
    """
    corpus = FastmriCorpus(_root(tmp_path), _annotations(tmp_path), _cache(tmp_path))
    destination = tmp_path / "labels" / "fastmri_findings.csv"

    corpus.labels(destination)

    table = pd.read_csv(destination).set_index(STUDY)
    assert table.index.tolist() == ["file100", "file101"]
    assert table.loc["file100", "ACL"] == POSITIVE_LEVEL
    assert table.loc["file101", "ACL"] == NEGATIVE_LEVEL, "a low-grade sprain is below our threshold"
    assert table.loc["file100", "Medial Meniscus"] == POSITIVE_LEVEL, "boxed in both compartments, so in this one"
    assert table.loc["file100", "Lateral Meniscus"] == POSITIVE_LEVEL, "and in the other one, symmetrically"
    assert table.loc["file101", "Medial Meniscus"] == NEGATIVE_LEVEL
    assert "Synovitis" not in table.columns, "the vocabulary has no synovial term; the readers were never asked"
    assert set(table["ACL__conf"]) == {READER_CONFIDENCE}


def test_straddled(tmp_path: Path):
    """A term boxed on BOTH sides of the compartment trough is located; one boxed on one side is not.

    Keyed like `read`, so a study that was listed and never boxed answers with an empty set rather than
    being absent — the same denominator rule, for the same reason.
    """
    straddling = FastmriCorpus(_root(tmp_path), _annotations(tmp_path), _cache(tmp_path)).straddled()

    assert set(straddling) == {"file100", "file101", "file102"}
    assert straddling["file100"] == frozenset({"Meniscus Tear"}), "boxed at 90 and at 240, either side of 165"
    assert straddling["file101"] == frozenset(), "one box cannot straddle"
    assert straddling["file102"] == frozenset(), "listed, never boxed"


def test_verdict_keeps_blocking_the_negative_when_a_coarse_term_sits_on_one_side():
    """The compartment rule ADDS a positive; it never turns a one-sided tear into a negative.

    A tear boxed in one compartment still proves a tear is present and still cannot say which compartment
    it is, which is the case `coarse` was written for and the case the geometry does not resolve.
    """
    reading = READINGS["Medial Meniscus"]
    seen = frozenset({"Meniscus Tear"})

    assert reading.verdict(seen, frozenset()) is None, "present, unlocated: no gradient either way"
    assert reading.verdict(seen, seen) == POSITIVE_LEVEL
    assert reading.verdict(frozenset(), frozenset()) == NEGATIVE_LEVEL


def test_verdict_refuses_the_compartment_rule_for_a_column_that_is_in_no_pair():
    """`PF OA` has no contralateral half, so straddling says nothing about it and must not assert.

    The cartilage columns are left non-compartmental for a different reason — the rule fires on 15 studies
    there, which is the n=4 MCL number one decimal place up — and this asserts the structural half.
    """
    straddling = frozenset(CARTILAGE_LOSS)

    assert READINGS["PF OA"].verdict(straddling, straddling) is None
    assert not READINGS["Medial OA"].compartmental, "15 studies is not a sample, it is a coincidence"


def test_labels_skips_a_volume_that_was_never_read(tmp_path: Path):
    """A study on disk but absent from the file list gets NO ROW — every column would be a guess.

    Unread means listed nowhere AND boxed nowhere: a box is itself a read, so a boxed study is admitted
    on the strength of the box even if the file list has not caught up with it.
    """
    annotations = _annotations(tmp_path, listed=("file100",), boxed=("file100",))
    corpus = FastmriCorpus(_root(tmp_path), annotations, _cache(tmp_path))
    destination = tmp_path / "labels" / "fastmri_findings.csv"

    corpus.labels(destination)

    assert pd.read_csv(destination)[STUDY].tolist() == ["file100"]
