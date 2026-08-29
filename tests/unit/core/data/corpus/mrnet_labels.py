import zipfile
from pathlib import Path

import pandas as pd
import pytest

from core.data.corpus.mrnet_labels import VALID_FIRST, ArchiveLabels, MRNetLabels, TableLabels

TRAIN_CASES = ("0000", "0001", "0002")
VALID_CASE = "1130"
ROOT = "MRNet-v1.0"


def _archive(tmp_path: Path) -> Path:
    """The Stanford layout: headerless, padded tables inside the zip."""
    path = tmp_path / "MRNet-v1.0.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{ROOT}/train-acl.csv", "".join(f"{case},{index % 2}\n" for index, case in enumerate(TRAIN_CASES))
        )
        archive.writestr(f"{ROOT}/valid-acl.csv", f"{VALID_CASE},1\n")
    return path


def _tables(tmp_path: Path) -> Path:
    """The portal layout: a separate directory, its own file names, unpadded cases and a header row."""
    directory = tmp_path / "tables"
    directory.mkdir(exist_ok=True)
    (directory / "mrnet-train-acl.csv").write_text("case,acl\n0,0\n1,1\n", encoding="utf-8")
    (directory / "mrnet-valid-acl.csv").write_text(f"case,acl\n{int(VALID_CASE)},1\n", encoding="utf-8")
    return directory


def test_case_of():
    """A case number is padded to the archive's width, and anything that is not a numeral is not a case."""
    assert MRNetLabels.case_of("7") == "0007", "an export that dropped the padding still has to join"
    assert MRNetLabels.case_of(" 0007 ") == "0007"
    assert MRNetLabels.case_of("case") is None, "the header row, dropped by reading it rather than declaring it"


def test_split_of():
    """The split is a function of the case number, which is what makes a flattened archive readable."""
    assert MRNetLabels.split_of("0000") == "train"
    assert MRNetLabels.split_of(str(VALID_FIRST - 1)) == "train"
    assert MRNetLabels.split_of(str(VALID_FIRST)) == "valid", "1130 is the first validation exam"


def test_verdicts():
    """Two columns become case-to-0/1, with the header dropped and the case padded on the way."""
    table = pd.DataFrame({0: ["case", "7", "1130"], 1: ["acl", "1", "0"]})

    assert MRNetLabels.verdicts(table) == {"0007": 1, "1130": 0}


def test_archive_labels_table(tmp_path: Path):
    """The Stanford tables read straight out of the zip, one split at a time."""
    assert ArchiveLabels(_archive(tmp_path)).table("train") == {"0000": 0, "0001": 1, "0002": 0}
    assert ArchiveLabels(_archive(tmp_path)).table("valid") == {VALID_CASE: 1}


def test_archive_labels_table_refuses_an_archive_without_the_split(tmp_path: Path):
    """The export that ships PIXELS ALONE has to fail here, so `--labels-dir` is reached for, not guessed."""
    bare = tmp_path / "bare.zip"
    with zipfile.ZipFile(bare, "w") as archive:
        archive.writestr("sagittal/0000.npy", b"")

    with pytest.raises(FileNotFoundError, match="train-acl.csv"):
        ArchiveLabels(bare).table("train")


def test_table_labels_table(tmp_path: Path):
    """The separate download answers identically, unpadded cases and header row and all.

    One contract, two sources: a caller picks by where its labels live, never by which reader it wants,
    and a difference between the two dicts would be the bug.
    """
    assert TableLabels(_tables(tmp_path)).table("train") == {"0000": 0, "0001": 1}
    assert TableLabels(_tables(tmp_path)).table("valid") == {VALID_CASE: 1}


def test_table_labels_table_refuses_a_directory_without_the_split(tmp_path: Path):
    """A missing table fails HERE rather than joining to nothing and reporting a clean 0.0%."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="valid"):
        TableLabels(empty).table("valid")
