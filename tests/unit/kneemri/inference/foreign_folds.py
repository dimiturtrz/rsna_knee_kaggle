from pathlib import Path

import pandas as pd

from core.findings import STUDY
from kneemri.inference.foreign_folds import ForeignFolds

KERNEL = """\
CROP_MM = 10_000.0

CROP_MM = 130.0


def render(value):
    return value * CROP_MM


def serve():
    return SERIES_ROOT
"""

SERVING = (3, 20)


def _folds(tmp_path: Path) -> ForeignFolds:
    kernel = tmp_path / "their_kernel.py"
    kernel.write_text(KERNEL, encoding="utf-8")
    return ForeignFolds(
        kernel=kernel,
        checkpoints=tmp_path / "ckpt",
        series_root=tmp_path / "series",
        symbols=("CROP_MM", "render", "serve"),
        scope=SERVING,
    )


def test_module(tmp_path: Path) -> None:
    """The lifted module carries the serving program's constants and the injected DICOM root."""
    module = _folds(tmp_path).module()
    assert module.render(2.0) == 260.0
    assert module.serve() == tmp_path / "series", "SERIES_ROOT came from their Kaggle mount; here it is ours"


def test_records(tmp_path: Path) -> None:
    """Their `by`: one entry per requested study, its series rows as dicts, everything else dropped."""
    series = pd.DataFrame(
        {
            STUDY: ["a", "a", "b"],
            "SeriesInstanceUID": ["s1", "s2", "s3"],
            "Anatomical_Plane": ["Sagittal", "Coronal", "Axial"],
        }
    )
    by = ForeignFolds.records(series, ["a"])
    assert set(by) == {"a"}, "a study we did not ask for is not decoded"
    assert [row["SeriesInstanceUID"] for row in by["a"]] == ["s1", "s2"]
