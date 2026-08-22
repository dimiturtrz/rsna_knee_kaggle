from pathlib import Path

import numpy as np
import pydicom

from core.data.corpus.diagnose_matrix import MatrixAudit
from tests.unit.conftest import write_series


def _mixed(directory: Path, odd: int) -> Path:
    """A series of eight slices where `odd` of them carry a different matrix size."""
    write_series(directory, slices=8, size=64)
    for name in sorted(directory.iterdir())[:odd]:
        data = pydicom.dcmread(str(name))
        data.Rows, data.Columns = 32, 32
        data.PixelData = np.zeros((32, 32), np.uint16).tobytes()
        data.save_as(str(name))
    return directory


def test_survey_one(tmp_path: Path):
    """A row per series, saying how many matrix sizes it holds and how many slices are the minority."""
    study = tmp_path / "study"
    write_series(study / "clean", slices=8, size=64)
    _mixed(study / "mixed", odd=2)

    rows = {str(row["series"]): row for row in MatrixAudit.survey_one(str(study))}

    assert rows["clean"]["distinct_shapes"] == 1
    assert rows["clean"]["minority_slices"] == 0
    assert rows["mixed"]["distinct_shapes"] == 2
    assert rows["mixed"]["minority_slices"] == 2, "the majority is what the series mostly is"
    assert rows["mixed"]["slices"] == 8


def test_run(tmp_path: Path):
    """Every study surveyed, one row per series, over the studies asked for and no others."""
    source = tmp_path / "series"
    for identifier in ("a", "b"):
        write_series(source / identifier / "sag", slices=4, size=32)
    write_series(source / "c" / "sag", slices=4, size=32)

    table = MatrixAudit().run(source, ["a", "b"], workers=2)

    assert len(table) == 2, "the study not asked for contributes nothing"
    assert set(table["study"]) == {"a", "b"}
    assert table["distinct_shapes"].tolist() == [1, 1]
