"""Between-slot crop offsets, on stacks whose right answer is known by construction."""

from pathlib import Path

import numpy as np
import pytest

from core.data.corpus.audit import AuditRequest
from core.data.corpus.crop_alignment import CropAlignment, SlotOffset, StudyOffsets
from core.data.corpus.study_cache import CorpusSample

SPACING = 2.0
BLOCK = 60


def _shifted_stack(down: int, across: int) -> np.ndarray:
    """A 200-pixel frame whose only tissue is a 60-pixel block, its corner placed where asked."""
    stack = np.zeros((3, 200, 200), np.float32)
    stack[:, down : down + BLOCK, across : across + BLOCK] = 1.0
    return stack


def test_magnitude():
    """A 3-4-5 triangle, so the value cannot be right by accident of one axis being zero."""
    assert SlotOffset("sag", 30.0, 40.0).magnitude() == 50.0
    assert SlotOffset("sag", 0.0, 0.0).magnitude() == 0.0


def test_offset():
    """A block centred 40 px past the middle reads as 40 px times the spacing, in millimetres."""
    alignment = CropAlignment()

    centred = alignment.offset(_shifted_stack(70, 70), SPACING, "sag")
    assert centred is not None
    assert abs(centred.down_mm) <= SPACING, "a block about the middle sits at the middle, to a pixel"
    assert abs(centred.across_mm) <= SPACING

    shifted = alignment.offset(_shifted_stack(110, 70), SPACING, "cor")
    assert shifted is not None
    assert shifted.plane == "cor"
    assert abs(shifted.down_mm - 80.0) <= SPACING, "40 px down at 2 mm/px"
    assert abs(shifted.across_mm) <= SPACING, "the axis it was not shifted on does not move"

    assert alignment.offset(_shifted_stack(70, 70), None, "sag") is None, "pixels do not compare between acquisitions"


def test_spread():
    """Slots that agree spread to zero; a slot that disagrees spreads by its own distance."""
    alignment = CropAlignment()
    agreeing = [SlotOffset("sag", 3.0, 4.0), SlotOffset("cor", 0.0, 5.0)]

    assert alignment.spread(agreeing) == 0.0, "equal magnitudes disagree by nothing, whatever their direction"
    assert alignment.spread([*agreeing, SlotOffset("ax", 0.0, 20.0)]) == 10.0, "mean of 0, 15, 15"
    assert alignment.spread([SlotOffset("sag", 3.0, 4.0)]) is None, "one slot has nothing to disagree with"
    assert alignment.spread([]) is None


def test_report():
    """The denominators lead, because a spread over what decoded is not a spread over the corpus."""
    study = [SlotOffset("sag", 3.0, 4.0), SlotOffset("cor", 0.0, 20.0)]
    lines = CropAlignment().report([study], CorpusSample(["a", "b"], 4407))

    assert lines[0].startswith("1 studies measured, from 2 studies sampled of 4407 in the corpus")
    assert "2 routed slots, 1 studies" in lines[0]
    assert any(line.startswith("sag") for line in lines)
    assert any(line.startswith("cor") for line in lines)
    assert any("within-study spread" in line and "p50 15.0 mm" in line for line in lines)

    empty = CropAlignment().report([], CorpusSample(["a"], 2))
    assert not any("within-study spread" in line for line in empty)


def test_plane_line():
    """One plane's row carries its magnitude tail and the signed components its axes make readable."""
    line = CropAlignment().plane_line("sag", [SlotOffset("sag", 3.0, -4.0), SlotOffset("sag", 3.0, -4.0)])

    assert line.startswith("sag")
    assert "5.0" in line, "the magnitude"
    assert "-4.0" in line, "the signed across component survives, unlike in the magnitude"
    assert line.endswith("      2")


def test_plane_lines():
    """One row per plane, in name order, each counting only its own plane's slots."""
    slots = [SlotOffset("sag", 3.0, -4.0), SlotOffset("cor", 0.0, 1.0), SlotOffset("sag", 3.0, -4.0)]
    lines = CropAlignment().plane_lines(slots)
    assert [line.split()[0] for line in lines] == ["cor", "sag"]
    assert lines[0].endswith("      1")
    assert lines[1].endswith("      2")
    assert CropAlignment().plane_lines([]) == []


def test_measure_study(tmp_path: Path):
    """A study that will not decode is silent about alignment rather than counted as aligned."""
    read = CropAlignment.measure_study(AuditRequest(identifier="absent", source=str(tmp_path), planes={}))

    assert len(read) == 1, "one study in, one study out — the pool flattens what a worker returns"
    assert read[0].offsets == []
    assert read[0].identifier == "absent", "the failure still names itself, or the join cannot tell who abstained"


def test_dump(tmp_path: Path):
    """Each study's spread is written beside its identifier, a one-slot study as nan rather than dropped."""
    destination = tmp_path / "spread.npz"
    CropAlignment().dump(
        destination,
        [
            StudyOffsets("pair", [SlotOffset("sag", 3.0, 4.0), SlotOffset("cor", 0.0, 20.0)]),
            StudyOffsets("lonely", [SlotOffset("sag", 3.0, 4.0)]),
        ],
    )
    written = np.load(destination, allow_pickle=True)

    assert written["studies"].tolist() == ["pair", "lonely"]
    assert written["spread"][0] == pytest.approx(15.0)
    assert np.isnan(written["spread"][1]), "a study with nothing to disagree with must abstain, not score zero"
    assert written["slots"].tolist() == [2, 1]
