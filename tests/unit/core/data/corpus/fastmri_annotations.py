"""The fastMRI+ annotation pair, written by the test in the shape the corpus ships it — footer included."""

from pathlib import Path

import numpy as np
import pandas as pd

from core.data.corpus.fastmri_annotations import FILE_LIST_FOOTER, FastmriAnnotations

LISTED = ("file100", "file101", "file102")


def _annotations(tmp_path: Path) -> Path:
    """Two boxed studies, a study-level row with no geometry, a read study with no box, a padded header.

    The padded ` label` is what `knee.csv` does to a reader that does not strip; the file list is headerless
    and ends in its `Knee data` footer.
    """
    path = tmp_path / "Annotations"
    path.mkdir()
    pd.DataFrame(
        [
            ["file100", 3, "No", 80.0, 20.0, 20.0, 6.0, "Meniscus Tear"],
            ["file100", 5, "No", 230.0, 20.0, 20.0, 6.0, "Meniscus Tear"],
            ["file100", 9, "No", 10.0, 2.0, 2.0, 2.0, "Joint Bodies"],
            ["file101", 5, "No", 12.0, 2.0, 8.0, 4.0, "Ligament - ACL High Grade Sprain"],
            ["file101", np.nan, "Yes", np.nan, np.nan, np.nan, np.nan, "artifact"],
        ],
        columns=["file", "slice", "study_level", "x", "y", "width", "height", " label"],
    ).to_csv(path / "knee.csv", index=False)
    pd.DataFrame({0: [*LISTED, FILE_LIST_FOOTER]}).to_csv(path / "knee_file_list.csv", index=False, header=False)
    return path


def test_read(tmp_path: Path):
    """Column names come back stripped, and the footer never enters the list of studies."""
    annotations = FastmriAnnotations.read(_annotations(tmp_path))

    assert "label" in annotations.boxes
    assert annotations.listed == LISTED, "the first line is a study; the last one is not"


def test_studies(tmp_path: Path):
    """The studies a reader opened: three, `Knee data` not among them."""
    assert FastmriAnnotations.read(_annotations(tmp_path)).studies() == frozenset(LISTED)


def test_terms(tmp_path: Path):
    """Every read study is keyed, the clean one with an empty set, and the footer is not a study."""
    seen = FastmriAnnotations.read(_annotations(tmp_path)).terms()

    assert set(seen) == set(LISTED)
    assert seen["file100"] == frozenset({"Meniscus Tear", "Joint Bodies"})
    assert seen["file101"] == frozenset({"Ligament - ACL High Grade Sprain", "artifact"}), "a study-level label reads"
    assert seen["file102"] == frozenset(), "read and clean, not unread"


def test_terms_admits_a_boxed_study_the_list_missed(tmp_path: Path):
    """A box is itself a read, so a study that carries one is keyed even if the list has not caught up."""
    directory = _annotations(tmp_path)
    pd.DataFrame({0: ["file100"]}).to_csv(directory / "knee_file_list.csv", index=False, header=False)

    assert set(FastmriAnnotations.read(directory).terms()) == {"file100", "file101"}


def test_straddling(tmp_path: Path):
    """A term boxed on both sides of the trough is located; one boxed once, or not at all, is not."""
    straddling = FastmriAnnotations.read(_annotations(tmp_path)).straddling()

    assert set(straddling) == set(LISTED)
    assert straddling["file100"] == frozenset({"Meniscus Tear"}), "centres at 90 and 240, either side of 165"
    assert straddling["file101"] == frozenset(), "one located box cannot straddle"
    assert straddling["file102"] == frozenset(), "listed, never boxed"


def test_located(tmp_path: Path):
    """Rows with geometry only, and only the columns a box placement reads."""
    located = FastmriAnnotations.read(_annotations(tmp_path)).located()

    assert len(located) == 4, "the study-level row has no box to place"
    assert list(located.columns) == ["file", "slice", "label", "x", "y", "width", "height"]
    assert located.index.tolist() == [0, 1, 2, 3]
