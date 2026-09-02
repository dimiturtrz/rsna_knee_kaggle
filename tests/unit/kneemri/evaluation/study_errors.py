import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.scores import LabelScores
from kneemri.evaluation.study_errors import StudyDamage, StudyErrors, TailProfile


def scrambled(studies: int) -> tuple[np.ndarray, np.ndarray]:
    """Alternating truth against a prediction that is neither perfect nor random, over all findings."""
    generator = np.random.default_rng(0)
    truth = np.tile((np.arange(studies) % 2).astype(np.float32)[:, None], (1, len(TARGETS)))
    predictions = truth.astype(np.float64) + generator.normal(0.0, 0.8, truth.shape)
    return predictions, truth


def test_discordant():
    """A perfect ranking charges nobody, a reversed one charges everybody, and a tie costs half."""
    positive = np.array([True, True, False, False])
    perfect = StudyErrors.discordant(np.array([1.0, 0.9, 0.2, 0.1]), positive)
    assert perfect.tolist() == [0.0, 0.0, 0.0, 0.0]

    reversed_order = StudyErrors.discordant(np.array([0.1, 0.2, 0.9, 1.0]), positive)
    assert reversed_order.tolist() == [1.0, 1.0, 1.0, 1.0]

    tied = StudyErrors.discordant(np.array([0.5, 0.5, 0.5, 0.5]), positive)
    assert tied.tolist() == [0.5, 0.5, 0.5, 0.5], "a tied pair is half discordant, as the rank AUC has it"

    absent = StudyErrors.discordant(np.array([1.0, 0.5]), np.array([False, False]))
    assert absent.tolist() == [0.0, 0.0], "a finding with no positive has no pairs to lose"


def test_damages():
    """The mean of a finding's column is exactly one minus that finding's AUC — the decomposition is exact."""
    predictions, truth = scrambled(60)
    table = StudyErrors().damages(predictions, truth)
    assert table.shape == predictions.shape
    for column, target in enumerate(TARGETS):
        area = LabelScores.area_under_curve(predictions[:, column], truth[:, column].astype(np.int64))
        assert table[:, column].mean() == pytest.approx(1.0 - area, abs=1e-9), f"{target} must decompose exactly"


def test_sided():
    """One class's mean share, and zero rather than a nan when the study has no cell of that class."""
    row = np.array([0.2, 0.4, 1.0])
    assert StudyErrors.sided(row, np.array([True, True, False])) == pytest.approx(0.3)
    assert StudyErrors.sided(row, np.array([False, False, False])) == 0.0


def test_ranked():
    """Worst study first, with its own worst finding named and its positive count carried."""
    predictions, truth = scrambled(40)
    studies = [f"study-{index}" for index in range(len(truth))]
    errors = StudyErrors()
    rows = errors.ranked(studies, errors.damages(predictions, truth), truth)
    assert len(rows) == len(studies)
    assert [row.damage for row in rows] == sorted((row.damage for row in rows), reverse=True)
    assert rows[0].worst_finding in TARGETS
    assert rows[0].worst_damage >= rows[0].damage, "the worst cell cannot sit below the mean over cells"
    assert {row.study for row in rows} == set(studies)


def test_direction():
    """A study losing its pairs on negative cells reads over-ranked, and on positive cells under-ranked."""
    over = StudyDamage("s", 0.3, 2, "ACL", 0.9, over=0.5, under=0.1)
    assert over.direction == "over"
    under = StudyDamage("s", 0.3, 2, "ACL", 0.9, over=0.1, under=0.5)
    assert under.direction == "under"


def test_report():
    """The tail lines are quoted, and a holdout too thin in gold refuses the label half outright."""
    thin = TailProfile(
        tail=90, body=819, tail_positives=4.4, body_positives=3.6, tail_gold=4, body_gold=9, tail_over=76, body_over=492
    )
    lines = thin.report()
    assert any("tail 90 studies" in line for line in lines)
    assert any("over-ranked" in line for line in lines)
    assert any("TOO FEW GOLD STUDIES" in line for line in lines), "13 gold cannot separate a tenth-sized tail"

    thick = TailProfile(
        tail=90,
        body=819,
        tail_positives=4.4,
        body_positives=3.6,
        tail_gold=20,
        body_gold=80,
        tail_over=76,
        body_over=492,
    )
    assert not any("TOO FEW GOLD" in line for line in thick.report()), "a populated gold set must be read, not refused"


def test_cut():
    """The cut takes a tenth, and both the gold and the direction counts land on the right side of it."""
    rows = [
        StudyDamage(
            f"study-{index}",
            damage=1.0 - index / 100.0,
            positives=index % 5,
            worst_finding="ACL",
            worst_damage=1.0,
            over=1.0 if index < 5 else 0.0,
            under=0.0 if index < 5 else 1.0,
        )
        for index in range(100)
    ]
    profile = TailProfile.cut(rows, gold={"study-0", "study-99"})
    assert (profile.tail, profile.body) == (10, 90)
    assert (profile.tail_gold, profile.body_gold) == (1, 1)
    assert (profile.tail_over, profile.body_over) == (5, 0)
