"""What makes two label tables the same ruler, and what makes them different ones."""

from pathlib import Path

from core.data.labels.ruler import DEFAULT_LABELS, STUDY_COLUMN, Ruler


def _table(path: Path, acl: list[float | None]) -> Ruler:
    """A minimal label table: one graded column with its two companions, one row per value."""
    rows = "\n".join(f"study{index},{'' if value is None else value},0.95,YES" for index, value in enumerate(acl))
    path.write_text(f"{STUDY_COLUMN},ACL,ACL__conf,ACL__verdict\n{rows}\n", encoding="utf-8")
    return Ruler(path)


def test_graded(tmp_path: Path):
    """Only the per-target values survive: the study id becomes the index, the companions are dropped."""
    graded = _table(tmp_path / DEFAULT_LABELS, [0.82, 0.28]).graded()

    assert list(graded.columns) == ["ACL"], "a __conf or __verdict column is not a truth an AUC reads"
    assert graded.index.name == STUDY_COLUMN
    assert len(graded) == 2


def test_moved_cells(tmp_path: Path):
    """A cell moves when it crosses 0.5 or gains or loses a value — a re-scale within one side does not.

    The three shapes the real tables take: `merged_labels.csv` is a column union that leaves our rows
    alone, `confident.csv` clears cells, and `report_labels_v3.csv` re-maps values. The last one is the
    case worth stating — a re-map that keeps every cell on its own side of the midpoint changes what is
    TRAINED on and not what is SCORED, so it is not a moved ruler and must not be reported as one.
    """
    ruler = _table(tmp_path / DEFAULT_LABELS, [0.82, 0.28, 0.08, 0.82])

    assert ruler.moved_cells(_table(tmp_path / "union.csv", [0.82, 0.28, 0.08, 0.82])) == 0
    assert ruler.moved_cells(_table(tmp_path / "cleared.csv", [0.82, None, None, 0.82])) == 2
    assert ruler.moved_cells(_table(tmp_path / "crossed.csv", [0.42, 0.28, 0.08, 0.82])) == 1
    assert ruler.moved_cells(_table(tmp_path / "rescaled.csv", [0.68, 0.13, 0.04, 0.68])) == 0, (
        "a re-map that keeps every cell on its own side of 0.5 moves the targets, not the ruler"
    )


def test_moved_cells_over_shared_studies_only(tmp_path: Path):
    """A table that only ADDS studies moves nothing: the comparison is over the rows they both carry.

    `merged_corpus` is why this is stated separately. Its table concatenates 736 foreign studies onto
    ours, and counting those as differences would have voided a roster member's 0.8545 for nothing.
    """
    ruler = _table(tmp_path / DEFAULT_LABELS, [0.82, 0.28])

    assert ruler.moved_cells(_table(tmp_path / "extended.csv", [0.82, 0.28, 0.91, 0.04])) == 0
