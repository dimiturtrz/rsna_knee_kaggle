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
