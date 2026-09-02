from pathlib import Path

import nibabel as nib
import numpy as np

from core.data.corpus.kmar_corpus import ARTIFACT_DIRECTORY, CLEAN_DIRECTORY, KmarCorpus, KmarPair
from core.data.corpus.kmar_stats import Measurement, Residual

MATRIX = 8


def _knee(background: float = 0.0, signal: float = 1.0) -> np.ndarray:
    """A slab of anatomy in the middle of an empty frame — enough to have an inside and an outside."""
    volume = np.full((2, MATRIX, MATRIX), background, dtype=np.float32)
    volume[:, 2:6, 2:6] = signal
    return volume


def test_scaled():
    """A volume is mapped onto [0, 1] by a window from ELSEWHERE, and clipped at both ends.

    Clipping is the point: the window comes from the clean volume, so a motion volume that overshoots it
    must be held at the boundary rather than renormalised, which is what would absorb the artifact.
    """
    volume = np.array([[[0.0, 0.5, 2.0]]], dtype=np.float32)

    assert Residual.scaled(volume, (0.0, 1.0)).tolist() == [[[0.0, 0.5, 1.0]]]
    assert Residual.scaled(volume, (1.0, 1.0)).max() == 0.0, "a degenerate window measures nothing"


def test_measured():
    """The three statistics of one pair, with the residual taken in the CLEAN volume's own window."""
    clean = _knee()
    artifact = clean.copy()
    artifact[:, 0, 0] = 1.0

    residual = Residual.measured(artifact, clean)

    assert residual is not None
    assert residual.residual > 0.0
    assert residual.background_share == 1.0, "the whole difference landed outside the knee"


def test_measured_separates_ghosting_from_blur():
    """Background share is what discriminates: motion deposits signal off the anatomy and blur does not.

    Two perturbations of the same size read the same on the residual mean and differently here, which is
    the only reason this statistic is carried at all.
    """
    clean = _knee()
    ghosted, blurred = clean.copy(), clean.copy()
    ghosted[:, 0, 0:2] = 1.0
    blurred[:, 3, 3:5] = 0.5

    ghost, blur = Residual.measured(ghosted, clean), Residual.measured(blurred, clean)

    assert ghost is not None and blur is not None
    assert ghost.background_share > blur.background_share, "ghosting lands outside the object, blur inside it"


def test_measured_refuses_a_pair_that_is_not_one_acquisition():
    """A shape mismatch returns None: the filenames matched and the arrays did not, so the join is wrong.

    Measuring it anyway would be impossible; measuring around it quietly would hide a broken join behind
    a plausible number.
    """
    assert Residual.measured(_knee()[:1], _knee()) is None


def test_summarised():
    """Each statistic at its 10th, 50th and 90th percentile over the pairs measured."""
    residuals = [Residual(residual=level, residual_p99=level * 2, background_share=0.5) for level in (0.0, 0.5, 1.0)]

    summary = Residual.summarised(residuals)

    assert set(summary) == {"residual", "residual_p99", "background_share"}
    assert summary["residual"]["p50"] == 0.5
    assert summary["residual_p99"]["p90"] == 1.8
    assert summary["background_share"] == {"p10": 0.5, "p50": 0.5, "p90": 0.5}


def _pair(tmp_path: Path, stem: str, plane: str, artifact: np.ndarray, clean: np.ndarray) -> KmarPair:
    """One pair on disk, written x, y, slice the way the corpus ships it."""
    paths = []
    for directory, volume in ((ARTIFACT_DIRECTORY, artifact), (CLEAN_DIRECTORY, clean)):
        target = tmp_path / f"{directory}_part1"
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{stem}.nii"
        nib.save(nib.Nifti1Image(np.transpose(volume, (1, 2, 0)), np.eye(4)), path)
        paths.append(path)
    return KmarPair(stem, paths[0], paths[1], plane, spacing=None)


def test_report(tmp_path: Path):
    """Pairs are grouped by plane, and a pair whose arrays disagree is COUNTED rather than dropped.

    The mismatch count is the denominator's honesty: a broken join that shrank the corpus silently would
    still produce a plausible per-plane summary, and nothing downstream could see the difference.
    """
    clean = _knee()
    ghosted = clean.copy()
    ghosted[:, 0, 0] = 1.0
    pairs = [
        _pair(tmp_path, "2020_1_sagittal_0.0", "sagittal", ghosted, clean),
        _pair(tmp_path, "2020_2_coronal_0.0", "coronal", ghosted, clean),
        _pair(tmp_path, "2020_3_coronal_0.0", "coronal", ghosted[:1], clean),
    ]

    report = Measurement(KmarCorpus(tmp_path)).report(pairs)

    assert report["measured"] == 2
    assert report["mismatched"] == 1, "the third pair's two halves are not one acquisition"
    planes = report["planes"]
    assert isinstance(planes, dict)
    assert sorted(planes) == ["coronal", "sagittal"]
    assert planes["coronal"]["pairs"] == 1, "the mismatched pair is not counted as a coronal measurement"
    assert planes["sagittal"]["background_share"]["p50"] == 1.0
