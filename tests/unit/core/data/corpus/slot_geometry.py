"""Slot placement in patient space, on headers whose geometry is known by construction."""

from pathlib import Path

import numpy as np
import pydicom

from core.data.corpus.acquisition import Acquisition
from core.data.corpus.audit import AuditRequest
from core.data.corpus.slot_geometry import SlotFrame, SlotGeometry, SlotPairing
from core.data.corpus.study_cache import CorpusSample

SAGITTAL = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
CORONAL = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
MATRIX = 200
SPACING = 0.5


def _header(orientation: tuple[float, ...], position: tuple[float, float, float]) -> pydicom.Dataset:
    """A 200x200 slice at 0.5 mm, placed and oriented as asked."""
    header = pydicom.Dataset()
    header.ImageOrientationPatient = list(orientation)
    header.ImagePositionPatient = list(position)
    header.PixelSpacing = [SPACING, SPACING]
    header.Rows = MATRIX
    header.Columns = MATRIX
    return header


def test_angle_to():
    """Orthogonal normals read 90 degrees, and a flipped scan direction is folded back to zero."""
    sagittal = SlotFrame("sag", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 120.0)

    assert sagittal.angle_to(SlotFrame("cor", (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 120.0)) == 90.0
    assert sagittal.angle_to(SlotFrame("sag", (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 120.0)) == 0.0, "sign is a convention"


def test_offset_to():
    """A 3-4-5 triangle in patient millimetres, so no single axis can carry the answer alone."""
    here = SlotFrame("sag", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 120.0)

    assert here.offset_to(SlotFrame("cor", (3.0, 4.0, 0.0), (0.0, 1.0, 0.0), 120.0)) == 5.0
    assert here.offset_to(here) == 0.0


def test_through_plane_to():
    """Only the component along this slot's own normal counts; the rest of the gap is invisible to it."""
    sagittal = SlotFrame("sag", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 120.0)

    assert sagittal.through_plane_to(SlotFrame("cor", (12.0, 40.0, 30.0), (0.0, 1.0, 0.0), 120.0)) == 12.0
    assert sagittal.through_plane_to(SlotFrame("cor", (-12.0, 0.0, 0.0), (0.0, 1.0, 0.0), 120.0)) == 12.0, "unsigned"
    assert sagittal.through_plane_to(SlotFrame("cor", (0.0, 99.0, 0.0), (0.0, 1.0, 0.0), 120.0)) == 0.0


def test_in_plane_to():
    """The two legs are a right triangle on the centre distance, so neither can absorb the other's share."""
    sagittal = SlotFrame("sag", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 120.0)
    other = SlotFrame("cor", (3.0, 4.0, 0.0), (0.0, 1.0, 0.0), 120.0)

    assert sagittal.in_plane_to(other) == 4.0, "5 mm total, 3 mm through"
    assert sagittal.in_plane_to(sagittal) == 0.0
    through, within = sagittal.through_plane_to(other), sagittal.in_plane_to(other)
    assert abs(through**2 + within**2 - sagittal.offset_to(other) ** 2) < 1e-9


def test_placed():
    """The corner walks to the matrix centre along the slice's own axes, and absent tags place nothing."""
    geometry = SlotGeometry()

    frame = geometry.placed(_header(SAGITTAL, (10.0, -50.0, 50.0)), "sag")
    assert frame is not None
    assert frame.plane == "sag"
    assert frame.centre == (10.0, 0.0, 0.0), "50 mm along +y and 50 mm along -z from the corner"
    assert frame.normal == (-1.0, 0.0, 0.0), "the cross product of the row and column cosines"

    assert geometry.placed(pydicom.Dataset(), "sag") is None, "no tags, no place"
    bare = _header(SAGITTAL, (0.0, 0.0, 0.0))
    del bare.Rows
    assert geometry.placed(bare, "sag") is None, "a matrix size is needed to reach the centre"


def test_triple():
    """A numpy row becomes plain floats, so a frame survives the pickle into a worker."""
    assert SlotGeometry().triple(np.array([1.5, -2.0, 3.0])) == (1.5, -2.0, 3.0)


def test_pairs():
    """Every pair once, carrying its angle and its centre distance."""
    frames = [
        SlotFrame("sag", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 120.0),
        SlotFrame("cor", (0.0, 3.0, 4.0), (0.0, 1.0, 0.0), 120.0),
        SlotFrame("ax", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 40.0),
    ]

    pairs = SlotPairing().pairs(frames)

    assert len(pairs) == 3, "three slots make three pairs"
    angle, total, through, within = pairs[0]
    assert (angle, total) == (90.0, 5.0)
    assert through == 0.0, "the sagittal normal is +x and the gap has no x component"
    assert within == 5.0, "so the whole displacement lies inside the sagittal plane"
    assert SlotPairing().pairs(frames[:1]) == [], "one slot makes no pair"


def test_frame():
    """A series with no files is silent about geometry rather than placed at the origin."""
    absent = Acquisition(
        study="s",
        series="1",
        directory=Path("nowhere"),
        files=(),
        weighting="PD",
        fat_suppressed=True,
        pixel_spacing=None,
        laterality_tag=None,
        centre_x=None,
    )

    assert SlotGeometry().frame(absent, "sag") is None


def test_report():
    """The denominators lead, and the orthogonal share is counted over the pairs actually formed."""
    study = [
        SlotFrame("sag", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 120.0),
        SlotFrame("cor", (0.0, 3.0, 4.0), (0.0, 1.0, 0.0), 120.0),
    ]

    lines = SlotPairing().report([study], CorpusSample(["a", "b"], 4407))

    assert lines[0].startswith("1 studies with geometry measured, from 2 studies sampled of 4407 in the corpus")
    assert "2 slots with geometry, 1 within-study slot pairs" in lines[0]
    assert any("normal angle between slots:   p50 90.0 deg" in line for line in lines)
    assert any("centre distance" in line and "p50 5.0 mm" in line for line in lines)
    assert any("through-plane leg" in line and "p50 0.0 mm" in line for line in lines)
    assert any("in-plane leg" in line and "p50 5.0 mm" in line for line in lines)
    assert any("orthogonal" in line and "1/1 (100.0%)" in line for line in lines)
    assert any("IMAGED the second's centre: 2/2 (100.0%)" in line for line in lines), "ordered, so both ways"
    assert lines == [
        "1 studies with geometry measured, from 2 studies sampled of 4407 in the corpus; "
        "2 slots with geometry, 1 within-study slot pairs",
        "normal angle between slots:   p50 90.0 deg, p90 90.0 deg",
        "centre distance        between slots: p50 5.0 mm, p90 5.0 mm",
        "  through-plane leg    between slots: p50 0.0 mm, p90 0.0 mm",
        "  in-plane leg         between slots: p50 5.0 mm, p90 5.0 mm",
        "pairs within 20 deg of orthogonal: 1/1 (100.0%)",
        "ordered pairs where the first slot IMAGED the second's centre: 2/2 (100.0%)",
    ], "pinned verbatim: the layout is what a reader compares across runs"

    unspanned = [SlotFrame(frame.plane, frame.centre, frame.normal, None) for frame in study]
    assert SlotPairing().report([unspanned], CorpusSample(["a", "b"], 4407)) == lines[:-1], (
        "no span, no coverage claim — the distributions stay and only the coverage line goes"
    )

    assert len(SlotPairing().report([], CorpusSample(["a"], 2))) == 1, "no pairs, no distributions"


def test_covers():
    """A centre inside half the stack's span is imaged; one beyond it is not; an unknown span says nothing."""
    sagittal = SlotFrame("sag", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 40.0)

    assert sagittal.covers(SlotFrame("cor", (19.0, 0.0, 0.0), (0.0, 1.0, 0.0), 120.0)) is True
    assert sagittal.covers(SlotFrame("cor", (21.0, 0.0, 0.0), (0.0, 1.0, 0.0), 120.0)) is False
    assert sagittal.covers(SlotFrame("cor", (0.0, 500.0, 0.0), (0.0, 1.0, 0.0), 120.0)) is True, "in-plane is free"
    assert SlotFrame("sag", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), None).covers(sagittal) is None


def test_coverage():
    """Ordered pairs, because a thick stack can image a thin one's centre without the reverse holding."""
    thick = SlotFrame("sag", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 140.0)
    thin = SlotFrame("ax", (50.0, 0.0, 0.0), (1.0, 0.0, 0.0), 20.0)

    assert SlotPairing().coverage([thick, thin]) == [True, False], "both directions, and they disagree"
    assert SlotPairing().coverage([thick]) == [], "one slot makes no ordered pair"
    assert SlotPairing().coverage([thick, SlotFrame("cor", (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), None)]) == [True]


def test_measure_study(tmp_path: Path):
    """A study that will not route is silent about geometry rather than counted as aligned."""
    assert SlotGeometry.measure_study(AuditRequest(identifier="absent", source=str(tmp_path), planes={})) == []
