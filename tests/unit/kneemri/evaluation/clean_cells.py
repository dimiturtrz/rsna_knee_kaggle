"""The mirror for the detector-clean re-cut: every public method called, and asserted."""

import csv
import math
from pathlib import Path

import numpy as np

from core.findings import STUDY, TARGETS
from kneemri.evaluation.clean_cells import TARGET_COLUMN, CleanCellRuler
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.runs.holdout_record import HoldoutReading, HoldoutRecord

STUDIES = ["s0", "s1", "s2", "s3"]
FLAGGED = [("s0", "ACL"), ("s1", "MCL")]


def _corrections(tmp_path: Path, cells: list[tuple[str, str]] = FLAGGED) -> Path:
    path = tmp_path / "corrections.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[STUDY, TARGET_COLUMN, "value"])
        writer.writeheader()
        for study, target in cells:
            writer.writerow({STUDY: study, TARGET_COLUMN: target, "value": "0.6"})
    return path


def _reading() -> HoldoutReading:
    truth = np.tile(np.array([[1.0], [0.0], [1.0], [0.0]], dtype=np.float32), (1, len(TARGETS)))
    predictions = np.tile(np.array([[0.9], [0.2], [0.8], [0.1]], dtype=np.float64), (1, len(TARGETS)))
    return HoldoutReading(studies=STUDIES, predictions=predictions, truth=truth, epoch=3, source="trained")


def _written(tmp_path: Path) -> Path:
    reading = _reading()
    destination = tmp_path / "member_s2026.pt"
    HoldoutRecord(destination).write(reading.studies, reading.predictions, reading.truth, reading.epoch)
    return HoldoutRecord(destination).path


def test_flagged_cells(tmp_path: Path) -> None:
    assert CleanCellRuler(_corrections(tmp_path)).flagged_cells() == frozenset(FLAGGED)


def test_mask(tmp_path: Path) -> None:
    """False exactly on the flagged cells, in the TARGETS column order and nowhere else."""
    selected = CleanCellRuler(_corrections(tmp_path)).mask(_reading())
    assert selected.shape == (len(STUDIES), len(TARGETS))
    assert not selected[0, TARGETS.index("ACL")]
    assert not selected[1, TARGETS.index("MCL")]
    assert selected[1, TARGETS.index("ACL")]
    assert int(np.count_nonzero(~selected)) == len(FLAGGED)


def test_per_label(tmp_path: Path) -> None:
    """The clean side drops a study from its own column alone; the flagged side reads only those cells."""
    ruler = CleanCellRuler(_corrections(tmp_path))
    clean = ruler.per_label(_reading(), LabelScores(MIDPOINT))
    assert clean["Effusion"] == 1.0
    assert clean["ACL"] == 1.0

    flagged = ruler.per_label(_reading(), LabelScores(MIDPOINT), clean=False)
    assert math.isnan(flagged["Effusion"])
    assert math.isnan(flagged["ACL"])


def test_macro(tmp_path: Path) -> None:
    ruler = CleanCellRuler(_corrections(tmp_path))
    assert ruler.macro(_reading(), LabelScores(MIDPOINT)) == 1.0


def test_row(tmp_path: Path) -> None:
    row = CleanCellRuler(_corrections(tmp_path)).row(_written(tmp_path))
    assert row.name == "member_s2026"
    assert row.clean_cells == len(STUDIES) * len(TARGETS) - len(FLAGGED)
    assert row.flagged_cells == len(FLAGGED)
    assert row.clean == 1.0
    assert row.clean75 == 1.0


def test_table(tmp_path: Path) -> None:
    """Named order is kept — this reads as a registered pair, never as a ranking."""
    path = _written(tmp_path)
    rows = CleanCellRuler(_corrections(tmp_path)).table([path, path])
    assert [row.name for row in rows] == ["member_s2026", "member_s2026"]
