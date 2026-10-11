"""The two guards that decide whether an annotated reading means anything: the refusal and the join."""

from pathlib import Path

import numpy as np
import pytest
import torch

from kneemri.evaluation.annotated_readings import AnnotatedReadings, TrainedOnAnnotationsError


def _reader(tmp_path: Path) -> AnnotatedReadings:
    """A reader over an empty root — neither guard touches pixels, a cache or a device."""
    from core.paths import DataRoot  # noqa: PLC0415 - test-local, so the module under test owns its imports

    return AnnotatedReadings(DataRoot(tmp_path), torch.device("cpu"))


def _checkpoint(path: Path, *, held: str) -> Path:
    """A checkpoint carrying only the options field the refusal reads."""
    torch.save({"options": {"hold_annotations": held}}, path)
    return path


def test_withheld(tmp_path: Path):
    """A run that held the annotations passes; one that weighted them is refused, not warned about."""
    reader = _reader(tmp_path)

    reader.withheld(_checkpoint(tmp_path / "honest.pt", held="True"))

    with pytest.raises(TrainedOnAnnotationsError, match="hold-annotations"):
        reader.withheld(_checkpoint(tmp_path / "trained.pt", held="False"))


def test_gold_fitted(tmp_path: Path):
    """Each of the three routes is named, and the one that records no flag is the one that mattered.

    `recal_dense` is the case: it passed `withheld`, recorded `calibrated_targets: False`, and carried the
    gold through the recalibrated table it was handed. A flag check alone clears it.
    """
    reader = _reader(tmp_path)

    def saved(name: str, **options: str) -> Path:
        path = tmp_path / name
        torch.save({"options": {"hold_annotations": "True", "labels": "/r/report_labels_v2.csv", **options}}, path)
        return path

    assert reader.gold_fitted(saved("clean.pt")) is None
    assert "MEASURED_AGREEMENT" in str(reader.gold_fitted(saved("calib.pt", calibrated_targets="True")))
    assert "per-finding" in str(reader.gold_fitted(saved("finding.pt", finding_calibration="True")))
    derived = reader.gold_fitted(saved("recal.pt", calibrated_targets="False", labels="/r/report_labels_v3.csv"))
    assert derived is not None, "a derived table carries the gold with every flag recording False"
    assert "report_labels_v3.csv" in derived


def test_origin(tmp_path: Path):
    """Only the members with a gold-fitted route are listed, under a caveat counting them; none, no caveat."""
    reader = _reader(tmp_path)
    clean = _checkpoint(tmp_path / "clean.pt", held="True")
    fitted = tmp_path / "fitted.pt"
    torch.save({"options": {"hold_annotations": "True", "calibrated_targets": "True"}}, fitted)

    assert reader.origin([clean]) == []
    lines = reader.origin([clean, fitted])
    assert lines[:2] == ["", "1 of them carry a constant FITTED on these cells, so their score here is agreement"]
    assert lines[-1].split()[0] == "fitted", "only the gold-fitted member is named"
    assert len(lines) == 4


def test_asked(tmp_path: Path):
    """No flag adds nothing; --per-label adds a blank line and the per-label table, nothing else."""
    from core.findings import TARGETS  # noqa: PLC0415 - test-local, so the module under test owns its imports
    from kneemri.evaluation.annotated_readings import AskedSections  # noqa: PLC0415 - test-local
    from kneemri.evaluation.scores import LabelScores  # noqa: PLC0415 - test-local

    reader = _reader(tmp_path)
    annotations = np.zeros((20, len(TARGETS)), dtype=np.float32)
    annotations[1::2, 0] = 1.0
    per_member = [(tmp_path / "best.pt", annotations.astype(np.float64))]
    sections = AskedSections(reader, LabelScores(), per_member, annotations)

    assert sections.asked(per_label=False, cross=False) == []
    table = reader.per_label_table(LabelScores(), per_member, annotations)
    assert sections.asked(per_label=True, cross=False) == ["", *table]
    crossed = sections.asked(per_label=True, cross=True)
    assert crossed[: len(table) + 1] == ["", *table], "the per-label section prints before the matrices"
    assert crossed[len(table) + 2].startswith("cross-column AUC, best")


def test_rows(tmp_path: Path):
    """The annotated studies located in CORPUS order, with a set that misses entirely refused."""
    reader = _reader(tmp_path)

    found = reader.rows(["s3", "s1", "s2", "s0"], ["s0", "s2"])

    assert found.dtype == np.int64
    assert found.tolist() == [2, 3], "corpus order, not the order the annotations were asked for"
    with pytest.raises(ValueError, match="none of the annotated studies"):
        reader.rows(["s3", "s1"], ["s0", "s2"])


def test_per_label_table(tmp_path: Path):
    """A finding too rare to read prints its COUNT and no number; a readable one prints both."""
    from core.findings import TARGETS  # noqa: PLC0415 - test-local, so the module under test owns its imports
    from kneemri.evaluation.scores import LabelScores  # noqa: PLC0415 - test-local

    reader = _reader(tmp_path)
    annotations = np.zeros((20, len(TARGETS)), dtype=np.float32)
    annotations[1::2, 0] = 1.0
    annotations[0, 1] = 1.0
    perfect = annotations.astype(np.float64)

    lines = reader.per_label_table(LabelScores(), [(tmp_path / "best.pt", perfect)], annotations)

    assert lines[0].startswith("finding"), "the header names the members whose columns follow"
    assert lines[1].split()[:2] == [TARGETS[0], "10"]
    assert "1.0000" in lines[1], "ten positives is readable, and a perfect ordering reads 1.0"
    assert "unreadable at n=20" in lines[2], "one positive is a count, never an AUC"
    assert lines[2].split()[:2] == [TARGETS[1], "1"], "the count is printed even where the AUC is withheld"


def test_cross(tmp_path: Path):
    """An off-diagonal cell reads ANOTHER finding's gold column, and a rare gold column is not a column.

    The equivalence classes are the two readings the grid exists to separate: a posterior that finds only
    its own gold column, and one that finds a neighbour's better than the neighbour's own head does.
    """
    from core.findings import TARGETS  # noqa: PLC0415 - test-local, so the module under test owns its imports
    from kneemri.evaluation.scores import LabelScores  # noqa: PLC0415 - test-local

    annotations = np.zeros((20, len(TARGETS)), dtype=np.float32)
    annotations[10:, 0] = 1.0
    annotations[10:, 1] = 1.0
    annotations[0, 2] = 1.0
    values = np.zeros((20, len(TARGETS)), dtype=np.float64)
    values[10:, 0] = 1.0

    lines = _reader(tmp_path).cross(LabelScores(), values, annotations)

    assert TARGETS[2][:7] not in lines[0], "one positive of twenty is a column no reading may use"
    assert lines[0].split()[-2:] == [TARGETS[0][:7], TARGETS[1][:7]], "the two readable gold columns, in order"
    assert lines[1].split()[-2:] == ["1.000", "1.000"], "the first posterior finds its own column AND its neighbour's"
    assert lines[3].split()[-2:] == ["0.500", "0.500"], "a flat posterior separates neither"


def test_paired(tmp_path: Path):
    """Every member is priced against the NAMED baseline, which never prices itself.

    The baseline row's absence is the whole guard: a table that reported the reference against itself
    would put a 0.0000 in the column the ordering is read from, and the row that names the pair is the
    one a reader trusts.
    """
    from core.findings import TARGETS  # noqa: PLC0415 - test-local, so the module under test owns its imports
    from kneemri.evaluation.scores import LabelScores  # noqa: PLC0415 - test-local

    reader = _reader(tmp_path)
    annotations = np.zeros((20, len(TARGETS)), dtype=np.float32)
    annotations[1::2] = 1.0
    perfect = annotations.astype(np.float64)
    inverted = 1.0 - perfect
    per_member = [(tmp_path / "best.pt", perfect), (tmp_path / "worst.pt", inverted)]

    lines = reader.paired(LabelScores(), per_member, tmp_path / "best.pt", annotations)

    assert "best" in lines[0], "the baseline is named on the header, not left to the reader"
    assert len(lines) == 2, "one row per member, and the baseline does not price itself"
    assert "worst" in lines[1]
    assert "-1.0000" in lines[1], "a perfectly inverted member is a full point below the baseline"


def test_joined(tmp_path: Path):
    """A saved foreign reading joins BY NAME, so a different study order does not cross the rows."""
    path = tmp_path / "foreign.npz"
    np.savez(path, identifiers=np.array(["c", "a", "b"]), values=np.array([[3.0], [1.0], [2.0]]))

    joined = _reader(tmp_path).joined([path], ["a", "b", "c"])

    assert len(joined) == 1
    assert joined[0][0] == path, "the file names the member, as a checkpoint names ours"
    assert np.allclose(joined[0][1], np.array([[1.0], [2.0], [3.0]])), "reordered onto the studies asked for"


def test_joined_missing(tmp_path: Path):
    """A reading that does not cover every study is refused rather than silently scored on a subset."""
    path = tmp_path / "short.npz"
    np.savez(path, identifiers=np.array(["a"]), values=np.array([[1.0]]))

    with pytest.raises(ValueError, match="holds no reading"):
        _reader(tmp_path).joined([path], ["a", "b"])


STUDY_NAMES = [f"s{index}" for index in range(12)]
PIN_HEAD = [
    "12 annotated studies, 3 member(s), none of them TRAINED on these cells",
    "m0                           0.1785",
    "m1                           0.2087",
    "m2                           0.2302",
    "combined                     0.0790",
    "",
]
PIN_ORIGIN = [
    "",
    "1 of them carry a constant FITTED on these cells, so their score here is agreement",
    "between two rulers that share an origin, never a confirmation (knee-vrioa):",
    "  m1                           --labels derived.csv",
]
PIN_PLAIN = [
    *PIN_HEAD,
    "paired against m0 (95% interval over studies, resampled):",
    "  m1                           +0.0302 [-0.0597, +0.1382] INSIDE the interval — no pairwise claim",
    "  m2                           +0.0517 [-0.0524, +0.1389] INSIDE the interval — no pairwise claim",
    *PIN_ORIGIN,
]
PIN_FLAGGED_LINES = 72
PIN_FLAGGED_DIGEST = "014dc11f0823794b26917d25d989634099115f507f53c75888643d454df371f9"
"""The 72-line flagged report, too long to inline, pinned by its sha256 off the module before the split."""


def _main_lines(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path, *flags: str):
    """Drive main with the device-bound read and the gold join stubbed, and return what it reported."""
    import logging  # noqa: PLC0415 - test-local

    from core import cli  # noqa: PLC0415 - test-local
    from core.findings import TARGETS  # noqa: PLC0415 - test-local
    from core.paths import DataRoot  # noqa: PLC0415 - test-local
    from kneemri.evaluation import annotated_readings  # noqa: PLC0415 - test-local
    from kneemri.evaluation.gold_rows import GoldRows  # noqa: PLC0415 - test-local

    truth = np.zeros((len(STUDY_NAMES), len(TARGETS)), dtype=np.float32)
    truth[1::2] = 1.0
    truth[0, 1] = 1.0
    rng = np.random.default_rng(3)
    members = [tmp_path / f"m{index}.pt" for index in range(3)]
    per_member = [
        (path, truth.astype(np.float64) + rng.normal(0, 0.6 + index / 4, truth.shape))
        for index, path in enumerate(members)
    ]
    monkeypatch.setattr(cli.Command, "rooted", lambda self, arguments: DataRoot(tmp_path))
    monkeypatch.setattr(GoldRows, "annotated", lambda self, table: (STUDY_NAMES, truth))
    monkeypatch.setattr(
        annotated_readings.AnnotatedReadings,
        "readings",
        lambda self, checkpoints, identifiers, *, overlap: (STUDY_NAMES[::-1], per_member),
    )
    monkeypatch.setattr(
        annotated_readings.AnnotatedReadings,
        "gold_fitted",
        lambda self, path: "--labels derived.csv" if path.stem == "m1" else None,
    )
    monkeypatch.setattr(
        "sys.argv", ["annotated_readings", "--device", "cpu", "--checkpoints", *map(str, members), *flags]
    )
    caplog.set_level(logging.INFO, logger=annotated_readings.__name__)
    annotated_readings.main()
    return [record.getMessage() for record in caplog.records if record.name == annotated_readings.__name__]


def test_main_reports_the_members_and_their_pairing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
):
    """Pinned verbatim off the module before main was split into report sections."""
    assert _main_lines(monkeypatch, caplog, tmp_path) == PIN_PLAIN


def test_main_adds_each_asked_section(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
):
    """--per-label and --cross each add their section, and a named baseline replaces the first checkpoint."""
    lines = _main_lines(monkeypatch, caplog, tmp_path, "--per-label", "--cross", "--against", str(tmp_path / "m2.pt"))
    import hashlib  # noqa: PLC0415 - test-local

    assert lines[: len(PIN_HEAD)] == PIN_HEAD
    assert lines[len(PIN_HEAD)] == "paired against m2 (95% interval over studies, resampled):"
    assert lines[-len(PIN_ORIGIN) :] == PIN_ORIGIN
    assert sum(line.startswith("cross-column AUC") for line in lines) == len(PIN_HEAD) - 3, "one matrix per member"
    assert len(lines) == PIN_FLAGGED_LINES
    assert hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest() == PIN_FLAGGED_DIGEST
