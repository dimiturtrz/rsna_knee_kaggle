import numpy as np

from kneemri.evaluation.ruler_check import RulerCheck
from kneemri.models.model import TARGETS

STUDIES = 40
SEED = 2026


def _predictions(strength: float) -> np.ndarray:
    """Scores that rank the first half above the second, the more so the larger `strength`."""
    generator = np.random.default_rng(SEED)
    noise = generator.random((STUDIES, len(TARGETS)))
    signal = np.zeros((STUDIES, len(TARGETS)))
    signal[: STUDIES // 2] = strength
    return signal + noise


def _truth() -> np.ndarray:
    """The first half positive on every finding."""
    truth = np.zeros((STUDIES, len(TARGETS)), dtype=np.float32)
    truth[: STUDIES // 2] = 1.0
    return truth


def test_under():
    """A ruler that drops pairs scores only the pairs it keeps, and still ranks the better model first."""
    check = RulerCheck()
    predictions = {"weak": _predictions(0.2), "strong": _predictions(1.5)}
    truth = _truth()

    full = check.under(predictions, truth)
    assert full["strong"] > full["weak"]

    # Drop the studies the weak model gets most wrong: its score under the narrower ruler must RISE,
    # which is what proves the selection is actually reaching the AUC rather than being ignored.
    keep = np.ones((STUDIES, len(TARGETS)), dtype=np.bool_)
    keep[STUDIES // 2 - 5 : STUDIES // 2 + 5] = False
    narrowed = check.under(predictions, truth, keep)

    assert narrowed["weak"] > full["weak"], "dropping the hardest pairs must change the number"
    assert narrowed["strong"] > narrowed["weak"], "the ordering must survive the change of ruler"


def test_report():
    """The verdict names agreement or disagreement rather than leaving it to be eyeballed."""
    agree = RulerCheck().report({"a": 0.83, "b": 0.81}, {"a": 0.87, "b": 0.84}, 0.73)
    assert any("RULERS AGREE" in line for line in agree)
    assert any("73.0%" in line for line in agree)

    flipped = RulerCheck().report({"a": 0.83, "b": 0.81}, {"a": 0.80, "b": 0.86}, 0.73)
    assert any("RULERS DISAGREE" in line for line in flipped)
    assert any("b > a" in line for line in flipped), "the disagreeing order has to be shown, not just flagged"
