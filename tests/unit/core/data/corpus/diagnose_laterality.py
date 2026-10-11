import math
from pathlib import Path

from core.data.corpus.acquisition import Acquisition
from core.data.corpus.diagnose_laterality import LateralityAudit, SideEvidence
from tests.unit.conftest import write_series


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


def _evidence(tmp_path: Path, *tags: str | None) -> SideEvidence:
    """One study with one sagittal series per tag given, probed the way the audit probes it."""
    study = tmp_path / "study"
    for index, tag in enumerate(tags):
        write_series(study / f"sag{index}", laterality=tag)
    return SideEvidence([Acquisition.probe("study", path.name, path) for path in sorted(study.iterdir())])


def test_tagged(tmp_path: Path):
    """Only the series that carry a tag contribute, in probe order."""
    assert _evidence(tmp_path, "R", None).tagged() == ["R"]


def test_centres(tmp_path: Path):
    """Every series whose header places it contributes a centre, tagged or not."""
    assert len(_evidence(tmp_path, "L", None).centres()) == 2


def test_geometric(tmp_path: Path):
    """Geometry ignores the tag entirely, so a tagged study still gets a geometric reading or none."""
    assert _evidence(tmp_path, "L").geometric() in ("L", "R", None)


def test_side(tmp_path: Path):
    """The tag wins where it exists."""
    assert _evidence(tmp_path, "L").side() == "L"


def test_source(tmp_path: Path):
    """A tag anywhere names the tag; otherwise geometry, or unresolved near the midline."""
    assert _evidence(tmp_path, "L").source() == "tag"
    assert _evidence(tmp_path / "bare", None).source() in ("geometry", "unresolved")


def test_row(tmp_path: Path):
    """An empty study fills every column with its empty value rather than raising."""
    row = SideEvidence([]).row()
    assert row["side"] == row["tag_side"] == row["geometry_side"] == ""
    assert row["source"] == "unresolved"
    assert row["series"] == 0
    assert math.isnan(float(str(row["median_centre_x"])))


def test_run(tmp_path: Path):
    """The sweep returns one row per study, which is what the holdout split is joined on."""
    source = tmp_path / "src"
    write_series(source / "a" / "sag", laterality="L")
    write_series(source / "b" / "sag", laterality="R")
    table = LateralityAudit().run(source, ["a", "b"], workers=1)
    assert len(table) == 2
    assert set(table["StudyInstanceUID"]) == {"a", "b"}
    assert set(table.columns) >= {"side", "source", "tag_side", "geometry_side", "series"}
