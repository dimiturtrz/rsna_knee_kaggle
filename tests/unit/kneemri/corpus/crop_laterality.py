"""Which side of the knee the in-plane crop discards."""

from pathlib import Path

import numpy as np

from kneemri.corpus.acquisition import Acquisition
from kneemri.corpus.crop_laterality import ClipRequest, CropLaterality, CropLateralityReport, SideClip

SPACING = 1.0
CROP_MM = 40.0
FRAME = 80


def _acquisition(column_axis_x: float | None) -> Acquisition:
    return Acquisition(
        study="s",
        series="1",
        directory=Path("."),
        files=(),
        weighting="PD",
        fat_suppressed=True,
        pixel_spacing=SPACING,
        laterality_tag=None,
        centre_x=None,
        column_axis_x=column_axis_x,
    )


def _stack(left_columns: int, right_columns: int) -> np.ndarray:
    """A frame with tissue only in the outer columns, so the crop discards a known amount per side."""
    stack = np.zeros((2, FRAME, FRAME), dtype=np.float32)
    stack[:, :, :left_columns] = 1.0
    stack[:, :, FRAME - right_columns :] = 1.0
    return stack


def test_lateral_column() -> None:
    """The header's x cosine and the knee's side multiply; either one missing declines the series."""
    measurement = CropLaterality(CROP_MM)
    # Columns advance toward the patient's left: on a LEFT knee that is toward lateral.
    assert measurement.lateral_column(_acquisition(1.0), "L") == 1
    assert measurement.lateral_column(_acquisition(1.0), "R") == -1
    assert measurement.lateral_column(_acquisition(-1.0), "L") == -1
    # A sagittal series carries no x in its column axis, and an unresolved side has no mapping at all.
    assert measurement.lateral_column(_acquisition(0.0), "L") is None
    assert measurement.lateral_column(_acquisition(None), "L") is None
    assert measurement.lateral_column(_acquisition(1.0), None) is None


def test_clipped() -> None:
    """Tissue outside the crop is attributed to the side the header says that edge is."""
    stack = _stack(left_columns=10, right_columns=5)

    high = CropLaterality(CROP_MM).clipped(stack, SPACING, lateral=1)
    low = CropLaterality(CROP_MM).clipped(stack, SPACING, lateral=-1)

    assert high is not None and low is not None
    # 15 tissue columns, 10 of them beyond the crop's low edge and 5 beyond its high edge.
    assert (round(high.lateral, 4), round(high.medial, 4)) == (round(5 / 15, 4), round(10 / 15, 4))
    assert (low.lateral, low.medial) == (high.medial, high.lateral)


def test_clipped_declines_what_the_crop_does_not_apply_to() -> None:
    """No spacing, and a frame already tighter than the crop, are silent rather than symmetric."""
    stack = _stack(left_columns=10, right_columns=5)

    assert CropLaterality(CROP_MM).clipped(stack, None, lateral=1) is None
    assert CropLaterality(1000.0).clipped(stack, SPACING, lateral=1) is None
    assert CropLaterality(CROP_MM).clipped(np.zeros_like(stack), SPACING, lateral=1) is None


def test_asymmetry() -> None:
    """Positive means the lateral side loses more."""
    assert SideClip(lateral=0.3, medial=0.1).asymmetry() == 0.3 - 0.1


def test_lines() -> None:
    """The denominators lead, and the verdict names symmetric as a refutation."""
    even = CropLateralityReport([SideClip(0.2, 0.2), SideClip(0.1, 0.1)], sampled=2, corpus=4407).lines()
    skewed = CropLateralityReport([SideClip(0.3, 0.05)] * 8, sampled=8, corpus=4407).lines()

    assert even[0].startswith("2 coronal/axial series measured, from 2 studies sampled of 4407")
    assert "SYMMETRIC" in even[1]
    assert "the crop clips one side harder" in skewed[1]


def test_lines_without_a_measurable_series_say_so_rather_than_dividing() -> None:
    """An empty sample is a denominator of zero, not a mean."""
    assert CropLateralityReport([], sampled=0, corpus=4407).lines() == [
        "0 coronal/axial series measured, from 0 studies sampled of 4407 in the corpus"
    ]


def test_measure_study() -> None:
    """A study that will not decode contributes nothing rather than a zero-zero clip."""
    assert CropLaterality.measure_study(ClipRequest("missing", ".", {})) == []
