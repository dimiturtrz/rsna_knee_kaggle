"""Acquired depth per routed slot, on rows whose distribution is known by construction."""

from pathlib import Path

from core.data.corpus.audit import AuditRequest
from core.data.corpus.native_depth import NativeDepth, SlotDepth
from core.data.corpus.study_cache import CorpusSample


def _depths(slot: str, slices: list[int], spacing: float = 3.5) -> list[SlotDepth]:
    return [SlotDepth(slot=slot, slices=count, spacing_mm=spacing) for count in slices]


def test_span_mm() -> None:
    """Slices times spacing — the extent the scanner actually imaged, not the extent a read asks for."""
    assert SlotDepth("SAG_T1", slices=30, spacing_mm=3.5).span_mm == 105.0


def test_row() -> None:
    """The tails are QUANTILES of the sample, so one unusually deep series cannot set the column."""
    # Nearest-rank at index n*q//100, so the eleventh value is p10 of a hundred — an outlier at either
    # end moves nothing, which is the whole reason the column is a quantile and not a min.
    row = NativeDepth().row("SAG_T1", _depths("SAG_T1", [*[20] * 11, *[30] * 79, *[400] * 10]))

    assert row.startswith("SAG_T1")
    fields = row.split()
    assert fields[1] == "100", "the count is this slot's series alone"
    assert (fields[2], fields[3], fields[4]) == ("20", "30", "400"), "p10, median, p90"
    assert fields[5] == "3.50"
    assert fields[6] == "105.0", "30 slices at 3.5 mm is the span the median series imaged"

    other = NativeDepth().row("COR_T1", [*_depths("SAG_T1", [99] * 4), *_depths("COR_T1", [30] * 4)])
    assert other.split()[1] == "4", "the other slot's series belong to the other slot"


def test_report() -> None:
    """The denominators lead, and each slot is one row — a depth over what routed is not a corpus depth."""
    measured = [*_depths("SAG_T1", [30] * 8), *_depths("COR_T1", [24] * 4, spacing=4.0)]

    lines = NativeDepth().report(measured, CorpusSample(["a", "b", "c"], 4407))

    assert lines[0] == "native depth: 12 series measured, from 3 studies sampled of 4407 in the corpus"
    assert "interpolates" in lines[1], "the reading the whole audit exists to separate is stated"
    assert sum(line.startswith("SAG_T1") for line in lines) == 1
    assert sum(line.startswith("COR_T1") for line in lines) == 1

    assert len(NativeDepth().report([], CorpusSample(["a"], 4407))) == 3, "no series, no rows"


def test_measure_study(tmp_path: Path) -> None:
    """A study that will not route is silent about depth rather than counted as a zero-slice series."""
    assert NativeDepth.measure_study(AuditRequest(identifier="absent", source=str(tmp_path), planes={})) == []
