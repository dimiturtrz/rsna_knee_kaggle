from pathlib import Path

from kneemri.corpus.diagnose_laterality import LateralityAudit

from ..conftest import write_series


def test_resolve_one(tmp_path: Path):
    """A tagged study reports the tag as its source, and geometry is computed alongside it anyway.

    Computing geometry even when a tag exists is the point: it is what makes "do the two agree?"
    answerable on the half of the corpus that has both.
    """
    study = tmp_path / "study0"
    write_series(study / "sag", laterality="L")
    row = LateralityAudit.resolve_one(("study0", str(study)))
    assert row["source"] == "tag"
    assert row["side"] == "L"
    assert row["tag_side"] == "L"
    assert row["geometry_side"] in ("L", "R", "")
    assert row["series"] == 1


def test_resolve_one_falls_back_to_geometry(tmp_path: Path):
    """With no tag anywhere, the side comes from the image centre — or stays unresolved near the midline."""
    study = tmp_path / "study1"
    write_series(study / "sag", laterality=None)
    row = LateralityAudit.resolve_one(("study1", str(study)))
    assert row["tag_side"] == ""
    assert row["source"] in ("geometry", "unresolved")


def test_run(tmp_path: Path):
    """The sweep returns one row per study, which is what the holdout split is joined on."""
    source = tmp_path / "src"
    write_series(source / "a" / "sag", laterality="L")
    write_series(source / "b" / "sag", laterality="R")
    table = LateralityAudit().run(source, ["a", "b"], workers=1)
    assert len(table) == 2
    assert set(table["StudyInstanceUID"]) == {"a", "b"}
    assert set(table.columns) >= {"side", "source", "tag_side", "geometry_side", "series"}
