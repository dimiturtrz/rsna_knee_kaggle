"""KMAR-50K read as studies: the subject grouping, the slot routing, and the mask they produce.

The equivalence classes here are the ones the real corpus contains and a pairs-shaped test cannot see:
a subject with one acquisition (918 of 1,101 on disk), a subject with several, a description naming a
slot these five cannot express, and a subject id that itself contains an underscore.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from core.data.corpus.kmar_corpus import ARTIFACT_DIRECTORY, CLEAN_DIRECTORY, KmarPair
from core.data.corpus.kmar_studies import KMAR_CORPUS, KmarStudies, KmarStudy
from core.data.corpus.slots import KMAR_SCHEME, KMAR_SLOTS, Plane, Routing
from core.data.corpus.study_cache import StudyCache

DEPTH, MATRIX = 4, 8
SPACING = 0.5

ACQUISITIONS: tuple[tuple[str, str], ...] = (
    # One subject read twice, in two planes, both fluid-weighted and fat-suppressed.
    ("2020_104_coronal_0.0", "pd_tse_fs_cor"),
    ("2020_104_transection_0.0", "qpd_tse_fs_tra_320"),
    # A subject read once, in T1 — the slot that proves the weighting axis is read and not assumed.
    ("2020_105_sagittal_0.0", "t1_tse_sag_p2"),
    # A subject id carrying an underscore of its own, which splitting on underscores would shift.
    ("2021_80-XXX_MR120_sagittal_0.0", "pd_tse_fs_sag"),
    # Coronal fluid-weighted WITHOUT fat suppression: a real meniscus sequence these five cannot hold.
    ("2020_106_coronal_0.0", "pd_tse_cor"),
)


def _write(directory: Path, stem: str) -> None:
    """One NIfTI written x, y, slice — the order the corpus ships and the reader has to transpose."""
    directory.mkdir(parents=True, exist_ok=True)
    pixels = np.linspace(0.0, 1.0, MATRIX * MATRIX * DEPTH, dtype=np.float32).reshape(MATRIX, MATRIX, DEPTH)
    nib.save(nib.Nifti1Image(pixels, np.eye(4)), directory / f"{stem}.nii")


def _root(tmp_path: Path) -> Path:
    """Both sides of the corpus plus a cohort table carrying a description for every volume."""
    root = tmp_path / "kmar50k"
    for stem, _ in ACQUISITIONS:
        _write(root / "KMAR-50K" / f"{ARTIFACT_DIRECTORY}_part1", stem)
        _write(root / "KMAR-50K" / f"{CLEAN_DIRECTORY}_part1", stem)
    pd.DataFrame(
        [
            {
                # The cohort writes the index as an int and a `.nii.gz` suffix; disk writes it as a float
                # and `.nii`. Same acquisition, two spellings — the join that has to survive.
                "NII_FileName": f"{stem.rsplit('_', 1)[0]}_0.nii.gz",
                "PixelSpacing": f" [{SPACING}, {SPACING}]",
                "SeriesDescription": f" '{description}'",
            }
            for stem, description in ACQUISITIONS
        ]
    ).to_csv(root / "KMAR-50K" / "TrainingCohort.csv", index=False)
    return root


def _studies(tmp_path: Path) -> KmarStudies:
    """A `KmarStudies` over a real cache directory, since `build` writes memmaps rather than arrays."""
    cache = StudyCache(tmp_path / "processed", slices=DEPTH, image_size=MATRIX, routing=Routing(scheme=KMAR_SCHEME))
    return KmarStudies(_root(tmp_path), cache)


def test_subject():
    """The plane and repeat index come off; an id containing an underscore survives intact."""
    assert KmarStudies.subject("2020_104_coronal_0.0") == "2020_104"
    assert KmarStudies.subject("2021_80-XXX_MR120_sagittal_0.0") == "2021_80-XXX_MR120", "the underscore held"
    assert KmarStudies.subject("nothing_like_a_volume") == "nothing_like_a_volume", "no plane word, no change"


def test_position():
    """Each description lands in the slot it names, and one that names none of the five answers None."""
    assert KMAR_SLOTS[KmarStudies.position("pd_tse_fs_cor") or 0].plane == Plane.CORONAL
    assert KMAR_SLOTS[KmarStudies.position("qpd_tse_fs_tra_320") or 0].plane == Plane.AXIAL, "tra is axial"
    sagittal_t1 = KmarStudies.position("t1_tse_sag_p2")
    assert sagittal_t1 is not None
    assert not KMAR_SLOTS[sagittal_t1].fluid_sensitive, "t1 is not fluid-weighted"
    assert KmarStudies.position("pd_tse_cor") is None, "fluid-weighted without fat suppression fits no slot"
    assert KmarStudies.position("") is None, "and an absent cohort row is not routed either"


def test_studies(tmp_path: Path):
    """Subjects are grouped, the unroutable volume is dropped, and each study keeps its spacing."""
    studies = _studies(tmp_path).studies()

    assert [study.identifier for study in studies] == ["2020_104", "2020_105", "2021_80-XXX_MR120"]
    assert len(studies[0].volumes) == 2, "one subject, two planes"
    assert len(studies[1].volumes) == 1
    assert studies[0].spacing == SPACING
    assert all(study.identifier != "2020_106" for study in studies), "the unroutable sequence is not a study"


def test_present(tmp_path: Path):
    """The mask is per-study rather than all-ones — the property that separates this from Rijeka."""
    studies = {study.identifier: study for study in _studies(tmp_path).studies()}

    assert sum(studies["2020_104"].present(len(KMAR_SLOTS))) == 2.0
    assert sum(studies["2020_105"].present(len(KMAR_SLOTS))) == 1.0, "most subjects fill exactly one"
    assert KmarStudy("x", {}, SPACING).present(len(KMAR_SLOTS)) == [0.0] * len(KMAR_SLOTS)


def test_metadata(tmp_path: Path):
    """One row per study, naming this corpus and the acquisition of its first filled slot."""
    rows = {row.identifier: row for row in _studies(tmp_path).metadata()}

    assert all(row.corpus == KMAR_CORPUS for row in rows.values())
    assert rows["2020_105"].sequence == "T1", "the T1 subject is not described as proton density"
    assert rows["2020_105"].fat_suppressed is False
    assert rows["2020_104"].pixel_spacing_mm == SPACING, "recorded, not derived — unlike every other corpus"


def test_build(tmp_path: Path):
    """The three artefacts land, and the mask holds the zeros the presence flags claimed."""
    studies = _studies(tmp_path)

    studies.build()

    assert studies.cache.images_path.exists()
    assert studies.cache.index_path.exists()
    mask = np.load(studies.cache.mask_path)
    assert mask.shape == (3, len(KMAR_SLOTS))
    assert mask.sum() == 4.0, "two slots on one subject and one on each of the others"


def test_paired():
    """The join's own denominator: a pair with no cohort row is unlisted, not unroutable."""
    listed = KmarPair("a", Path("a"), Path("a"), "sagittal", SPACING, "pd_tse_fs_sag")
    unlisted = KmarPair("b", Path("b"), Path("b"), "sagittal", None, "")

    assert KmarStudies.paired([listed, unlisted]) == 1


def test_studies_reports_an_empty_corpus(tmp_path: Path):
    """An empty root is an empty list rather than a crash — the state a half-finished fetch leaves."""
    cache = StudyCache(tmp_path / "out", slices=DEPTH, image_size=MATRIX, routing=Routing(scheme=KMAR_SCHEME))

    assert KmarStudies(tmp_path / "nothing", cache).studies() == []


@pytest.mark.parametrize("description", ["pd_tse_fs_sag", "t1_tse_cor", "qpd_tse_fs_tra"])
def test_position_is_a_canonical_subsequence(description: str):
    """Every routed position indexes `KMAR_SLOTS`, which `MergedCache` copies by NAME in this order."""
    position = KmarStudies.position(description)

    assert position is not None
    assert 0 <= position < len(KMAR_SLOTS)
