from pathlib import Path

import numpy as np
import pandas as pd

from kneemri.model import TARGETS
from kneemri.targets import ANNOTATED_WEIGHT, Targets


def _tables(tmp_path: Path) -> tuple[Path, Path]:
    """A derived table for three studies and an annotation table that covers only the first."""
    derived: dict[str, list[object]] = {"StudyInstanceUID": ["a", "b", "c"]}
    for target in TARGETS:
        derived[target] = [0.9, 0.1, 0.5]
        derived[f"{target}__conf"] = [1.0, 0.25, 0.0]
    derived_path = tmp_path / "report_labels.csv"
    pd.DataFrame(derived).to_csv(derived_path, index=False)

    annotated: dict[str, list[object]] = {"StudyInstanceUID": ["a"], "Report": ["x"]}
    for target in TARGETS:
        annotated[target] = [0.0]
    annotated_path = tmp_path / "train.csv"
    pd.DataFrame(annotated).to_csv(annotated_path, index=False)
    return derived_path, annotated_path


def test_load(tmp_path: Path):
    """Derived scores carry their confidence as a weight; an image annotation overrides and outweighs."""
    derived_path, annotated_path = _tables(tmp_path)
    targets = Targets.load(derived_path, annotated_path, ["a", "b", "c"])
    assert targets.scores.shape == (3, len(TARGETS))
    assert targets.scores[0, 0] == 0.0, "the annotation overrides the report-derived score"
    assert targets.weights[0, 0] == ANNOTATED_WEIGHT, "and outweighs every derived row"
    assert targets.scores[1, 0] == 0.1
    assert targets.weights[1, 0] == 0.25
    assert targets.weights[2, 0] == 0.0, "a finding the report never mentions pulls with weight zero"


def test_rows(tmp_path: Path):
    """A batch of positions returns its scores and weights aligned."""
    derived_path, annotated_path = _tables(tmp_path)
    targets = Targets.load(derived_path, annotated_path, ["a", "b", "c"])
    scores, weights = targets.rows(np.array([2, 1], dtype=np.int64))
    assert scores.shape == (2, len(TARGETS))
    assert weights[0, 0] == 0.0
    assert weights[1, 0] == 0.25


def test_coverage(tmp_path: Path):
    """Coverage is the share of studies carrying any weight — where the reading is thin shows here."""
    derived_path, annotated_path = _tables(tmp_path)
    coverage = Targets.load(derived_path, annotated_path, ["a", "b", "c"]).coverage()
    assert set(coverage) == set(TARGETS)
    assert coverage["ACL"] == 2 / 3
