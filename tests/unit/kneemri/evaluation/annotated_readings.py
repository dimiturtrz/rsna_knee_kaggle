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
