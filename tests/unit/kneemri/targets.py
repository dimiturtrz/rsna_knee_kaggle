from pathlib import Path

import numpy as np
import pandas as pd

from kneemri.model import TARGETS
from kneemri.targets import ANNOTATED_WEIGHT, CONFIDENCE_FLOOR, Targets


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


def test_load_applies_a_confidence_floor(tmp_path: Path):
    """With a floor, a finding the report never mentions still pulls — weakly, not not at all."""
    derived_path, annotated_path = _tables(tmp_path)
    floored = Targets.load(derived_path, annotated_path, ["a", "b", "c"], CONFIDENCE_FLOOR)
    dropped = Targets.load(derived_path, annotated_path, ["a", "b", "c"])
    assert dropped.weights[2, 0] == 0.0, "without a floor an unmentioned finding is dropped"
    assert floored.weights[2, 0] == CONFIDENCE_FLOOR, "with a floor it keeps a quarter of the weight"
    # a fully-confident derived target is unchanged by the floor
    assert floored.weights[1, 0] > dropped.weights[1, 0]


def test_pair(tmp_path: Path):
    """Training and scoring tables are separable, so two arms can be judged with one ruler."""
    derived_path, annotated_path = _tables(tmp_path)
    other = tmp_path / "other.csv"
    frame = pd.read_csv(derived_path)
    for target in TARGETS:
        frame[target] = 1.0 - frame[target]
    frame.to_csv(other, index=False)

    training, evaluation = Targets.pair(derived_path, other, annotated_path, ["a", "b", "c"])
    assert training.scores[1, 0] != evaluation.scores[1, 0], "the ruler must be the OTHER table"

    same_training, same_evaluation = Targets.pair(derived_path, None, annotated_path, ["a", "b", "c"])
    assert same_training is same_evaluation, "no eval table means score on what you trained on"
