from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data.labels.targets import (
    ANNOTATED_WEIGHT,
    ANNOTATIONS,
    CALIBRATED,
    CONFIDENCE_FLOOR,
    FINDING,
    FINDING_OFFSETS,
    SEVERITY_WEIGHT,
    WEIGHT_FLOOR,
    Supervision,
    Targets,
    UncalibratedFindingShiftError,
)
from core.findings import TARGETS


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
    floored = Targets.load(derived_path, annotated_path, ["a", "b", "c"], Supervision(floor=CONFIDENCE_FLOOR))
    dropped = Targets.load(derived_path, annotated_path, ["a", "b", "c"])
    assert dropped.weights[2, 0] == 0.0, "without a floor an unmentioned finding is dropped"
    assert floored.weights[2, 0] == CONFIDENCE_FLOOR, "with a floor it keeps a quarter of the weight"
    # a fully-confident derived target is unchanged by the floor
    assert floored.weights[1, 0] > dropped.weights[1, 0]


def test_supervised(tmp_path: Path):
    """A loaded table answers for the switches it was built under, so the pre-flight check can ask it.

    Asserted through `load` rather than the constructor because the path under test is the one that
    defaults: a `load` that dropped its argument would report the plain recipe on a calibrated run.
    """
    derived_path, annotated_path = _tables(tmp_path)
    supervision = Supervision(floor=CONFIDENCE_FLOOR, calibrated=True)

    loaded = Targets.load(derived_path, annotated_path, ["a", "b", "c"], supervision)

    assert loaded.supervised() == {
        WEIGHT_FLOOR: CONFIDENCE_FLOOR,
        CALIBRATED: True,
        FINDING: False,
        SEVERITY_WEIGHT: False,
        ANNOTATIONS: True,
    }
    assert Targets.load(derived_path, annotated_path, ["a"]).supervised() == {
        WEIGHT_FLOOR: 0.0,
        CALIBRATED: False,
        FINDING: False,
        SEVERITY_WEIGHT: False,
        ANNOTATIONS: True,
    }


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


def _levelled(tmp_path: Path) -> tuple[Path, Path]:
    """A derived table whose values are the levels the reader actually emits."""
    derived: dict[str, list[object]] = {"StudyInstanceUID": ["a", "b", "c"]}
    for target in TARGETS:
        derived[target] = [0.68, 0.82, 0.94]
        derived[f"{target}__conf"] = [0.95, 0.95, 0.95]
    derived_path = tmp_path / "levelled.csv"
    pd.DataFrame(derived).to_csv(derived_path, index=False)
    annotated_path = tmp_path / "empty_train.csv"
    pd.DataFrame({"StudyInstanceUID": [], "Report": []}).to_csv(annotated_path, index=False)
    return derived_path, annotated_path


def test_applied():
    """Each emitted level becomes what it was measured to be worth; anything else is left alone.

    The pass-through is the part that matters. Snapping an unrecognised value to a nearest level would
    silently reinterpret a table on a different scale as if it were this one - and these values were
    themselves assigned rather than measured, which is the whole reason this class exists.
    """
    calibration = Supervision(calibrated=True)
    value = np.array([0.08, 0.28, 0.68, 0.82, 0.94, 0.5, np.nan], dtype=np.float32)

    applied = calibration.applied(value)

    assert applied[:5].tolist() == pytest.approx([0.041, 0.136, 0.378, 0.688, 0.958], abs=1e-6)
    assert applied[5] == pytest.approx(0.5), "a value the reader never emits passes through"
    assert np.isnan(applied[6]), "a missing cell stays missing rather than becoming a rate"
    assert value[0] == pytest.approx(0.08), "the input must not be mutated"


def test_reliability():
    """A denial is graded on being right about ABSENCE, so its rate is read from the other side.

    Derived from the same table `applied` spends on the value, which is the point: one measurement, two
    uses, and no second copy of the numbers to fall out of date.
    """
    reliability = dict(Supervision().reliability())

    assert reliability[0.08] == pytest.approx(1.0 - 0.041), "a NO is right when the finding is absent"
    assert reliability[0.68] == pytest.approx(0.378)
    assert reliability[0.82] == pytest.approx(0.688)
    assert reliability[0.94] == pytest.approx(0.958)
    assert 0.28 not in reliability, "the level that asserts nothing has no hit rate to compute"


def test_trusted():
    """Severity sets the weight; without the switch the reader's own confidence is untouched."""
    value = np.array([0.08, 0.28, 0.68, 0.82, 0.94, 0.5], dtype=np.float32)
    confidence = np.full(6, 0.95, dtype=np.float64)

    graded = Supervision(severity=True).trusted(value, confidence)

    assert graded.tolist() == pytest.approx([0.959, 0.95, 0.378, 0.688, 0.958, 0.95], abs=1e-6)
    assert graded[2] < graded[3] < graded[4], "the three YES grades must separate, which is the mechanism"
    assert confidence[2] == pytest.approx(0.95), "the input must not be mutated"
    assert Supervision().trusted(value, confidence).tolist() == pytest.approx(confidence.tolist())


def test_moves_the_ruler():
    """Either label switch forces the holdout to be re-read plain; a floor alone does not."""
    assert not Supervision().moves_the_ruler()
    assert not Supervision(floor=CONFIDENCE_FLOOR).moves_the_ruler(), "applied identically to arm and control"
    assert Supervision(calibrated=True).moves_the_ruler()
    assert Supervision(severity=True).moves_the_ruler()
    assert Supervision(calibrated=True, finding=True).moves_the_ruler()
    assert Supervision(annotations=False).moves_the_ruler(), "696 cells move back to what the report said"


def test_per_finding():
    """A finding's cells move by its own offset, in logit space, and only when the switch is on."""
    supervision = Supervision(calibrated=True, finding=True)
    value = np.array([0.378, np.nan], dtype=np.float32)

    over_called = supervision.per_finding(value, "Contusion")
    under_called = supervision.per_finding(value, "Synovitis")

    assert over_called[0] < 0.378, "a finding the reader over-calls trains at less than the pooled rate"
    assert under_called[0] > 0.378, "one it misses in silence trains at more"
    assert np.isnan(over_called[1]), "a missing cell stays missing rather than becoming a rate"
    assert Supervision(calibrated=True).per_finding(value, "Contusion")[0] == pytest.approx(0.378), (
        "without the switch the pooled table is reproduced exactly"
    )
    assert value[0] == pytest.approx(0.378), "the input must not be mutated"


def test_per_finding_names_exactly_the_findings_it_shifts():
    """An offset for a finding that does not exist would be silently ignored, which is how a typo hides."""
    assert tuple(name for name, _ in FINDING_OFFSETS) == tuple(TARGETS)


def test_finding_calibration_refuses_to_shift_an_uncalibrated_table():
    """The offsets are fitted ON TOP of the pooled level map — applied to raw levels they are nonsense."""
    with pytest.raises(UncalibratedFindingShiftError):
        Supervision(finding=True)


def test_load_grades_the_weight_without_moving_the_target(tmp_path: Path):
    """The whole mechanism in one assertion: same scores, three different weights where there was one."""
    labels, train_table = _levelled(tmp_path)
    identifiers = ["a", "b", "c"]

    graded = Targets.load(labels, train_table, identifiers, Supervision(floor=CONFIDENCE_FLOOR, severity=True))
    control = Targets.load(labels, train_table, identifiers, Supervision(floor=CONFIDENCE_FLOOR))

    assert graded.scores.tolist() == control.scores.tolist(), "values are untouched — this is not a calibration"
    assert control.weights[0, 0] == pytest.approx(control.weights[2, 0]), "mild and severe pull alike today"
    assert graded.weights[0, 0] == pytest.approx(0.25 + 0.75 * 0.378, abs=1e-6)
    assert graded.weights[2, 0] == pytest.approx(0.25 + 0.75 * 0.958, abs=1e-6)
    assert graded.weights[0, 0] < graded.weights[1, 0] < graded.weights[2, 0]


def test_pair_leaves_the_ruler_ungraded(tmp_path: Path):
    """Grading the ruler too would change which holdout cells count, so the arm must not touch it."""
    labels, train_table = _levelled(tmp_path)

    training, ruler = Targets.pair(labels, None, train_table, ["a", "b", "c"], Supervision(severity=True))

    assert training is not ruler, "one object for both would grade the holdout with the targets"
    assert training.weights[0, 0] == pytest.approx(0.378), "the training table carries the graded weight"
    assert ruler.weights[0, 0] == pytest.approx(0.95), "the ruler keeps the reader's own confidence"


def test_pair_leaves_the_ruler_uncalibrated(tmp_path: Path):
    """Calibrating the ruler too would raise every arm's AUC by construction, so it must not happen."""
    labels, train_table = _levelled(tmp_path)
    identifiers = ["a", "b", "c"]

    training, ruler = Targets.pair(labels, None, train_table, identifiers, Supervision(calibrated=True))

    assert training is not ruler, "one object for both would recalibrate the ruler with the targets"
    assert training.scores[0, 0] == pytest.approx(0.378), "the training table carries the measured rate"
    assert ruler.scores[0, 0] == pytest.approx(0.68), "the ruler keeps the value it always had"


def test_load_can_withhold_the_annotations_so_a_member_never_trains_on_them(tmp_path: Path):
    """The one input `supervision_gap` needs: a study whose radiologist labels the loss never read.

    Study "a" is annotated 0.0 against a report score of 0.9, so the two are distinguishable by value
    as well as by weight — a guard that dropped only the WEIGHT would leave the annotation's score in
    the table and the member would still be fitted on it.
    """
    derived_path, annotated_path = _tables(tmp_path)
    identifiers = ["a", "b", "c"]

    held = Targets.load(derived_path, annotated_path, identifiers, Supervision(annotations=False))
    control = Targets.load(derived_path, annotated_path, identifiers)

    assert control.scores[0, 0] == 0.0, "the control trains on the annotation"
    assert held.scores[0, 0] == pytest.approx(0.9), "the arm trains on what the REPORT said instead"
    assert held.weights[0, 0] == pytest.approx(1.0), "and at the report's own confidence, not 3.0"
    assert held.supervised()[ANNOTATIONS] is False, "recorded, so the pre-flight check proves it fired"
    assert held.scores[1, 0] == control.scores[1, 0], "an unannotated study is untouched either way"


def test_pair_keeps_the_annotations_in_the_ruler_when_the_arm_withholds_them(tmp_path: Path):
    """What makes the instrument comparable to every ledger row: only the TRAINING table changes.

    Withholding them from the ruler too would move this arm's holdout onto a different table from the
    one every other member was scored on, which is the exact incomparability `pair` exists to prevent.
    """
    derived_path, annotated_path = _tables(tmp_path)

    training, ruler = Targets.pair(derived_path, None, annotated_path, ["a", "b", "c"], Supervision(annotations=False))

    assert training is not ruler, "one object for both would hide the annotations from the ruler too"
    assert training.scores[0, 0] == pytest.approx(0.9), "the arm is fitted on the report score"
    assert ruler.scores[0, 0] == 0.0, "the ruler keeps the annotation every other member was scored on"
    assert ruler.weights[0, 0] == ANNOTATED_WEIGHT


def test_pair_without_a_calibration_still_shares_one_table(tmp_path: Path):
    """No calibration means no second read: the ruler IS the training table, as it always was."""
    labels, train_table = _levelled(tmp_path)

    training, ruler = Targets.pair(labels, None, train_table, ["a", "b", "c"])

    assert training is ruler
