"""The metadata schema's mirror: every public method called, and the two spellings held together."""

from pathlib import Path

import pandas as pd

from core.data.corpus.merged_cache import CORPUS as MERGED_CORPUS
from core.data.corpus.study_metadata import CORPUS, IDENTIFIER, StudyMetadata

RIJEKA = StudyMetadata("rijeka-0001", "rijeka", plane="Sagittal", fat_suppressed=True, slice_spacing_mm=0.5)
BARE = StudyMetadata("canonical-0001", "canonical")


def test_axes() -> None:
    axes = StudyMetadata.axes()
    assert "plane" in axes
    assert "scanner" in axes
    assert IDENTIFIER not in axes
    assert CORPUS not in axes


def test_row() -> None:
    row = RIJEKA.row()
    assert row[IDENTIFIER] == "rijeka-0001"
    assert row[CORPUS] == "rijeka"
    assert row["plane"] == "Sagittal"
    assert row["age_years"] is None


def test_frame() -> None:
    table = StudyMetadata.frame([RIJEKA, BARE])
    assert list(table.columns) == [IDENTIFIER, CORPUS, *StudyMetadata.axes()]
    assert len(table) == 2
    assert table["scanner"].isna().all()
    assert table.loc[table[IDENTIFIER] == "rijeka-0001", "slice_spacing_mm"].iloc[0] == 0.5


def test_frame_keeps_columns_when_empty() -> None:
    table = StudyMetadata.frame([])
    assert list(table.columns) == [IDENTIFIER, CORPUS, *StudyMetadata.axes()]
    assert table.empty


def test_joined() -> None:
    labels = pd.DataFrame({IDENTIFIER: ["rijeka-0001", "unknown-0002"], "ACL": [1.0, 0.0]})
    joined = StudyMetadata.joined(labels, StudyMetadata.frame([RIJEKA, BARE]))
    assert len(joined) == 2
    assert joined.loc[joined[IDENTIFIER] == "rijeka-0001", "plane"].iloc[0] == "Sagittal"
    assert pd.isna(joined.loc[joined[IDENTIFIER] == "unknown-0002", "plane"].iloc[0])


def test_joined_drops_the_duplicated_corpus_column() -> None:
    labels = pd.DataFrame({IDENTIFIER: ["rijeka-0001"], CORPUS: ["rijeka"], "ACL": [1.0]})
    joined = StudyMetadata.joined(labels, StudyMetadata.frame([RIJEKA]))
    assert list(joined.columns).count(CORPUS) == 1
    assert joined[CORPUS].iloc[0] == "rijeka"


def test_write(tmp_path: Path) -> None:
    destination = tmp_path / "metadata" / "rijeka_metadata.csv"
    StudyMetadata.write([RIJEKA, BARE], destination)
    written = pd.read_csv(destination)
    assert list(written.columns) == [IDENTIFIER, CORPUS, *StudyMetadata.axes()]
    assert written[IDENTIFIER].tolist() == ["rijeka-0001", "canonical-0001"]
    assert written["plane"].iloc[0] == "Sagittal"
    assert written["plane"].isna().iloc[1], "a corpus that cannot answer reads absent, not empty-string"


def test_merged(tmp_path: Path) -> None:
    first = tmp_path / "rijeka_metadata.csv"
    StudyMetadata.write([RIJEKA], first)
    stale = tmp_path / "old_metadata.csv"
    # A corpus written before an axis existed, carrying one this schema never declared.
    pd.DataFrame({IDENTIFIER: ["old-0001"], CORPUS: ["old"], "coil_channels": [15]}).to_csv(stale, index=False)

    merged = StudyMetadata.merged([first, stale])
    assert list(merged.columns) == [IDENTIFIER, CORPUS, *StudyMetadata.axes()]
    assert merged[CORPUS].tolist() == ["rijeka", "old"]
    assert "coil_channels" not in merged.columns, "an axis reaches a consumer only by being declared"
    assert pd.isna(merged.loc[merged[IDENTIFIER] == "old-0001", "plane"].iloc[0])


def test_merge_written(tmp_path: Path) -> None:
    """The named corpora that wrote a file are concatenated; one that wrote none is skipped, not fatal."""
    StudyMetadata.write([RIJEKA], tmp_path / "rijeka_metadata.csv")
    StudyMetadata.write([BARE], tmp_path / "canonical_metadata.csv")

    StudyMetadata.merge_written(tmp_path, ("rijeka", "canonical", "absent"), "merged")

    written = pd.read_csv(tmp_path / "merged_metadata.csv")
    assert written[IDENTIFIER].tolist() == ["rijeka-0001", "canonical-0001"]
    assert list(written.columns) == [IDENTIFIER, CORPUS, *StudyMetadata.axes()]


def test_merge_written_writes_nothing_when_no_source_did(tmp_path: Path) -> None:
    StudyMetadata.merge_written(tmp_path, ("absent",), "merged")
    assert not (tmp_path / "merged_metadata.csv").exists()


def test_corpus_column_matches_the_merge_that_writes_it() -> None:
    assert CORPUS == MERGED_CORPUS
