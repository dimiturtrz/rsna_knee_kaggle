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


def test_rows(tmp_path: Path):
    """The annotated studies located in CORPUS order, with a set that misses entirely refused."""
    reader = _reader(tmp_path)

    found = reader.rows(["s3", "s1", "s2", "s0"], ["s0", "s2"])

    assert found.dtype == np.int64
    assert found.tolist() == [2, 3], "corpus order, not the order the annotations were asked for"
    with pytest.raises(ValueError, match="none of the annotated studies"):
        reader.rows(["s3", "s1"], ["s0", "s2"])


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
