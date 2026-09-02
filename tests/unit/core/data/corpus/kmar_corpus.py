from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from core.data.corpus.kmar_corpus import ARTIFACT_DIRECTORY, CLEAN_DIRECTORY, KmarCorpus

DEPTH, MATRIX = 3, 8
SPACING = 0.703125

STEMS = ("2020_1_transection_0.0", "2020_2_sagittal_0.0", "2020_2_sagittal_1.0")


def _write(directory: Path, stem: str, level: float) -> None:
    """One NIfTI written x, y, slice — the order the corpus ships and the reader has to transpose."""
    directory.mkdir(parents=True, exist_ok=True)
    pixels = np.full((MATRIX, MATRIX, DEPTH), level, dtype=np.float32)
    nib.save(nib.Nifti1Image(pixels, np.eye(4)), directory / f"{stem}.nii")


def _root(tmp_path: Path, missing: str = "") -> Path:
    """Two part directories, plus a cohort table one level down from where the volumes are found.

    `missing` drops one stem from the clean side, which is the real state of this corpus — 1,339 artifact
    volumes against 1,341 clean ones — and the case a join that trusted the two sides to match would get
    silently wrong.
    """
    root = tmp_path / "kmar50k"
    for stem in STEMS:
        _write(root / "KMAR-50K" / f"{ARTIFACT_DIRECTORY}_part1", stem, level=0.4)
        if stem != missing:
            _write(root / "KMAR-50K" / f"{CLEAN_DIRECTORY}_part1", stem, level=0.5)
    cohort = pd.DataFrame(
        [
            # The cohort writes the index as an int and a `.nii.gz` suffix; disk writes it as a float and
            # `.nii`. Same acquisition, two spellings — the join that has to survive.
            {"NII_FileName": "2020_1_transection_0.nii.gz", "PixelSpacing": f" [{SPACING}, {SPACING}]"},
            {"NII_FileName": "2020_2_sagittal_0.nii.gz", "PixelSpacing": f" [{SPACING}, {SPACING}]"},
        ]
    )
    cohort.to_csv(root / "KMAR-50K" / "TrainingCohort.csv", index=False)
    return root


def test_pairs(tmp_path: Path):
    """Only stems present on BOTH sides are paired, and each carries its plane and its cohort spacing."""
    pairs = KmarCorpus(_root(tmp_path, missing=STEMS[2])).pairs()

    assert [pair.identifier for pair in pairs] == [STEMS[0], STEMS[1]], "the clean side is missing the third"
    assert [pair.plane for pair in pairs] == ["transection", "sagittal"]
    assert pairs[0].artifact.parent.name.startswith(ARTIFACT_DIRECTORY)
    assert pairs[0].clean.parent.name.startswith(CLEAN_DIRECTORY)
    assert pairs[0].spacing == SPACING


def test_pairs_carries_a_pair_the_cohort_never_listed(tmp_path: Path):
    """A pair with no metadata row keeps its spacing None rather than being dropped.

    The fetch left more volumes on disk than either cohort table lists, so dropping them would quietly
    shrink the corpus a measurement runs over — and report a rate over a denominator nobody chose.
    """
    pairs = {pair.identifier: pair for pair in KmarCorpus(_root(tmp_path)).pairs()}

    assert pairs[STEMS[2]].spacing is None, "the cohort lists index 0 for this subject, not index 1"
    assert pairs[STEMS[1]].spacing == SPACING


def test_spacings(tmp_path: Path):
    """The cohort tables are found one level down, and `PixelSpacing` is parsed as the literal it is."""
    spacings = KmarCorpus(_root(tmp_path)).spacings()

    assert spacings == {"2020_1_transection_0": SPACING, "2020_2_sagittal_0": SPACING}


def test_volume(tmp_path: Path):
    """A NIfTI arrives slices-first, which is the axis order every other corpus here is read in.

    Getting this wrong would hand a statistic about the in-plane phase-encode direction a volume whose
    second axis is the slice axis, and nothing downstream could see it.
    """
    root = _root(tmp_path)
    corpus = KmarCorpus(root)

    volume = corpus.volume(root / "KMAR-50K" / f"{ARTIFACT_DIRECTORY}_part1" / f"{STEMS[0]}.nii")

    assert volume.shape == (DEPTH, MATRIX, MATRIX)
    assert volume.dtype == np.float32
    assert abs(float(volume.mean()) - 0.4) < 1e-6
