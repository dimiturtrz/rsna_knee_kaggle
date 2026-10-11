from collections import Counter
from pathlib import Path

import pytest

from core.data.corpus.slot_ties import ContestedCensus, SlotTies, SlotTiesReport, UnroutedCensus
from tests.unit.conftest import write_series

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


def test_contested():
    """The report gives a rate per slot and states what a low count would and would not settle."""
    lines = SlotTiesReport(Counter({"SAG_FLUID_FS": 20, "studies_with_any": 25}), STUDIES).contested()
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


def test_unrouted():
    """The census names the sequences the scheme cannot express, and refuses to read as a verdict."""
    totals = Counter({"Coronal|fluid|nofs": 1139, "Axial|t1|nofs": 621, "__series__": 24371, "__slices__": 83297})
    lines = SlotTiesReport(totals, STUDIES).unrouted()
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


def test_unrouted_census_worker():
    """The unrouted census hands the pool the worker that names lost sequences."""
    assert UnroutedCensus().worker() is SlotTies._lost_of


def test_contested_census_worker():
    """The contested census hands the pool the worker that counts tied slots."""
    assert ContestedCensus().worker() is SlotTies._contested_of


def test_unrouted_census_tally():
    """Each study's non-zero counts are summed in; a zero adds no key, so no bookkeeping row is invented."""
    totals: Counter[str] = Counter()
    for counted in ({"Axial|t1|nofs": 2, "__series__": 7}, {"Axial|t1|nofs": 1, "Coronal|fluid|fs": 0}):
        UnroutedCensus().tally(totals, counted)
    assert totals == Counter({"Axial|t1|nofs": 3, "__series__": 7})
    assert "studies_with_any" not in totals


def test_contested_census_tally():
    """A study with any tied slot counts once toward studies_with_any; an all-zero study adds nothing."""
    totals: Counter[str] = Counter()
    for counted in ({"SAG_FLUID_FS": 2, "COR_T1": 1}, {"SAG_FLUID_FS": 0}):
        ContestedCensus().tally(totals, counted)
    assert totals == Counter({"SAG_FLUID_FS": 2, "COR_T1": 1, "studies_with_any": 1})


def test_unrouted_census_lines():
    """The unrouted census renders as the report's unrouted section."""
    report = SlotTiesReport(Counter({"Axial|t1|nofs": 3, "__series__": 10, "__slices__": 40}), 2)
    assert UnroutedCensus().lines(report) == report.unrouted()


def test_contested_census_lines():
    """The contested census renders as the report's contested section."""
    report = SlotTiesReport(Counter({"SAG_FLUID_FS": 1, "studies_with_any": 1}), 2)
    assert ContestedCensus().lines(report) == report.contested()


def _main_lines(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path, *flags: str
) -> list[str]:
    """Drive main over three study directories with the DICOM-reading workers stubbed by name."""
    import concurrent.futures  # noqa: PLC0415 - test-local
    import logging  # noqa: PLC0415 - test-local

    from core import cli  # noqa: PLC0415 - test-local
    from core.data.corpus import slot_ties  # noqa: PLC0415 - test-local
    from core.data.corpus.slots import SLOTS  # noqa: PLC0415 - test-local
    from core.data.corpus.study_cache import StudyCache  # noqa: PLC0415 - test-local

    (tmp_path / "train_series.csv").write_text(
        "StudyInstanceUID,SeriesInstanceUID\nb,1.1\nb,1.2\nc,2.9\nc,2.1\n", encoding="utf-8"
    )
    for name in ("b", "c", "a"):
        (tmp_path / "train_series" / name).mkdir(parents=True)
    contested = {"a": {SLOTS[0].name: 1, SLOTS[1].name: 0}, "b": {SLOTS[0].name: 0}, "c": {SLOTS[2].name: 2}}
    lost = {
        "a": Counter({"Axial|t1|nofs": 2, "__slices__": 40, "__series__": 7}),
        "b": Counter({"Coronal|fluid|fs": 1, "Axial|t1|nofs": 1, "__slices__": 30, "__series__": 6}),
        "c": Counter({"__slices__": 0, "__series__": 5}),
    }

    class Rooted:
        def raw(self) -> Path:
            return tmp_path

    monkeypatch.setattr(cli.Command, "rooted", lambda self, arguments: Rooted())
    monkeypatch.setattr(StudyCache, "planes_of", staticmethod(lambda table: {}))
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)
    monkeypatch.setattr(SlotTies, "_contested_of", staticmethod(lambda argument: contested[argument[0].name]))
    monkeypatch.setattr(SlotTies, "_lost_of", staticmethod(lambda argument: lost[argument[0].name]))
    monkeypatch.setattr("sys.argv", ["slot_ties", "--workers", "2", *flags])
    caplog.set_level(logging.INFO, logger=slot_ties.__name__)
    slot_ties.main()
    return [record.getMessage() for record in caplog.records if record.name == slot_ties.__name__]


PIN_ORDER = "1 of 2 studies order their series differently in the table than on disk (50.0%)"
PIN_VERDICTS = [
    "  a LOW number kills the tie-break hypothesis for the 0.011 leaderboard gap - the two pipelines would agree"
    " almost everywhere, and the residual must come from elsewhere",
    "  a HIGH number does not prove it either: a tie only matters when the tied series differ in what they show."
    " That is the next measurement, not this one",
]
PIN_QUIET = [
    "  COR_FLUID_FS: 0 (0.0% of studies)",
    "  AX_FLUID_FS: 0 (0.0% of studies)",
    "  SAG_FLUID_NOFS: 0 (0.0% of studies)",
    "  COR_T1: 0 (0.0% of studies)",
    "  SAG_T1: 0 (0.0% of studies)",
]
"""The five slots no stubbed study contests; the reports were pinned verbatim off main before it was split."""


def test_main_counts_contested_slots(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path):
    """The order verdict first, then the contested census over the studies asked for."""
    assert _main_lines(monkeypatch, caplog, tmp_path, "--studies", "2") == [
        PIN_ORDER,
        "contested slots over 2 studies:",
        "  SAG_FLUID_FS: 1 (50.0% of studies)",
        *PIN_QUIET,
        "  studies with at least one contested slot: 1 (50.0%)",
        *PIN_VERDICTS,
    ]


def test_main_counts_unrouted_series(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path):
    """--lost swaps the worker and the table, and no study is counted as contested."""
    assert _main_lines(monkeypatch, caplog, tmp_path, "--lost") == [
        PIN_ORDER,
        "3 studies, 18 series acquired, 4 unrouted (22.2%)",
        "  Axial|t1|nofs: 3",
        "  Coronal|fluid|fs: 1",
        "  slices in unrouted series: 70",
        "  a discarded localiser is worth nothing and a discarded axial T1 may be worth a lot - the SEQUENCE"
        " breakdown is the answer here, not the percentage",
    ]


def test_main_names_a_table_in_sorted_order(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
):
    """A table already in sorted-UID order says so, because that is the verdict that refutes the hypothesis."""
    (tmp_path / "test_series.csv").write_text("StudyInstanceUID,SeriesInstanceUID\nb,1.1\nb,1.2\n", encoding="utf-8")
    (tmp_path / "test_series" / "b").mkdir(parents=True)
    assert _main_lines(monkeypatch, caplog, tmp_path, "--split", "test_series") == [
        "0 of 1 studies order their series differently in the table than on disk (0.0%)"
        " — SAME ORDER, so both tie rules pick the same series everywhere",
        "contested slots over 1 studies:",
        "  SAG_FLUID_FS: 0 (0.0% of studies)",
        *PIN_QUIET,
        "  studies with at least one contested slot: 0 (0.0%)",
        *PIN_VERDICTS,
    ]
