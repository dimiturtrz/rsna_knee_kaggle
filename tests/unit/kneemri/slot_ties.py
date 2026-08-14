from collections import Counter
from pathlib import Path

from kneemri.slot_ties import SlotTies

from .conftest import write_series

STUDIES = 200


def test_of_study(tmp_path: Path):
    """A study whose two sagittal fluid series have equal slice counts reports that slot contested."""
    study = tmp_path / "study"
    for name in ("a", "b"):
        write_series(study / name, slices=12)
    planes = {"a": "Sagittal", "b": "Sagittal"}

    contested = SlotTies().of_study(study, planes)

    assert contested["SAG_FLUID_FS"] == 1
    assert contested["COR_T1"] == 0, "a slot no series matches cannot be contested"


def test_of_study_is_clear_when_one_series_is_denser(tmp_path: Path):
    """The tie-break is unambiguous when a single candidate leads, whatever the list order."""
    study = tmp_path / "study"
    write_series(study / "thin", slices=8)
    write_series(study / "thick", slices=20)

    contested = SlotTies().of_study(study, {"thin": "Sagittal", "thick": "Sagittal"})

    assert contested["SAG_FLUID_FS"] == 0


def test_report():
    """The report gives a rate per slot and states what a low count would and would not settle."""
    lines = SlotTies().report(Counter({"SAG_FLUID_FS": 20, "studies_with_any": 25}), STUDIES)
    text = "\n".join(lines)

    assert "SAG_FLUID_FS: 20 (10.0% of studies)" in text
    assert "at least one contested slot: 25 (12.5%)" in text
    assert "kills the tie-break hypothesis" in text
    assert "does not prove it either" in text, "a high count must not be reported as a confirmation"


def test_lost_of_study(tmp_path: Path):
    """An axial T1 has no slot to land in, so it is counted by the axes that failed to match."""
    study = tmp_path / "study"
    write_series(study / "ax_t1", slices=10, description="AX T1", repetition=500.0, echo=10.0, scan_options="")
    write_series(study / "sag", slices=12)

    lost = SlotTies().lost_of_study(study, {"ax_t1": "Axial", "sag": "Sagittal"})

    assert lost["Axial|t1|nofs"] == 1
    assert lost["__slices__"] == 10, "the imaging that goes with a lost series is what decides if it matters"
    assert lost["__series__"] == 2, "the denominator counts everything acquired, routed or not"


def test_lost_report():
    """The census names the sequences the scheme cannot express, and refuses to read as a verdict."""
    totals = Counter({"Coronal|fluid|nofs": 1139, "Axial|t1|nofs": 621, "__series__": 24371, "__slices__": 83297})
    lines = SlotTies().lost_report(totals, STUDIES)
    text = "\n".join(lines)

    assert "1760 unrouted (7.2%)" in text, "the share counts only real sequence keys, not the bookkeeping ones"
    assert "Coronal|fluid|nofs: 1139" in text
    assert "slices in unrouted series: 83297" in text
    assert "a discarded localiser is worth nothing" in text


def test_disagreeing_order(tmp_path: Path):
    """Counts only the studies whose table order differs from the sorted-UID order we route by.

    The real corpus answers ZERO, which is what refutes the tie-break hypothesis — so the test builds
    both cases explicitly rather than trusting a fixture that happens to agree.
    """
    table = tmp_path / "series.csv"
    table.write_text(
        "StudyInstanceUID,SeriesInstanceUID\n"
        "agrees,1.1\nagrees,1.2\n"  # already sorted, as the whole real corpus is
        "differs,2.9\ndiffers,2.1\n",  # the table leads with the LATER uid
        encoding="utf-8",
    )

    disagreeing, total = SlotTies().disagreeing_order(table)

    assert (disagreeing, total) == (1, 2)
