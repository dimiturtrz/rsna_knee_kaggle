"""Prevalence matching: the rich finding re-scored at the poor finding's counts."""

import numpy as np

from kneemri.evaluation.prevalence_match import MatchedGap, PrevalenceMatch
from kneemri.models.model import TARGETS

RICH = "Medial Meniscus"
POOR = "Lateral Meniscus"
DRAWS = 32


def _corpus(rich_separation: float, poor_separation: float) -> tuple[np.ndarray, np.ndarray]:
    """A corpus where the rich label has many positives and the poor one few, with set orderings."""
    studies = 200
    predictions = np.zeros((studies, len(TARGETS)), dtype=np.float32)
    truth = np.zeros((studies, len(TARGETS)), dtype=np.float32)

    for target, positives, separation in ((RICH, 80, rich_separation), (POOR, 20, poor_separation)):
        column = TARGETS.index(target)
        truth[:positives, column] = 1.0
        # Positives score above negatives on `separation` of the studies, so the AUC is set by construction.
        ranked = np.linspace(0.0, 1.0, studies, dtype=np.float32)
        predictions[:positives, column] = ranked[:positives] * (1.0 - separation) + separation
        predictions[positives:, column] = ranked[positives:] * (1.0 - separation)
    return predictions, truth


def test_counted() -> None:
    """Positives and negatives are counted at the project's midpoint cut."""
    _, truth = _corpus(0.9, 0.9)
    assert PrevalenceMatch().counted(truth, POOR) == (20, 180)
    assert PrevalenceMatch().counted(truth, RICH) == (80, 120)


def test_matched() -> None:
    """Both labels are drawn at the shared counts — the smaller of each side — and differenced."""
    predictions, truth = _corpus(0.9, 0.3)
    gap = PrevalenceMatch(resamples=DRAWS).matched(predictions, truth, RICH, POOR)

    # 20 positives (the poor label's) and 120 negatives (the rich label's): neither label supplies both.
    assert (gap.positives, gap.negatives) == (20, 120)
    low, median, high = gap.difference
    assert low <= median <= high
    # Matching removes sample size, not the separation the two labels were built with.
    assert median > 0.0
    assert gap.poor_auc < gap.rich_auc


def test_matched_is_seeded() -> None:
    """Two runs of the same comparison return the same interval, so a quoted number is recomputable."""
    predictions, truth = _corpus(0.9, 0.3)
    first = PrevalenceMatch(resamples=DRAWS).matched(predictions, truth, RICH, POOR)
    second = PrevalenceMatch(resamples=DRAWS).matched(predictions, truth, RICH, POOR)
    assert first.difference == second.difference


def test_explained() -> None:
    """A gap is explained by prevalence when the matched difference straddles zero."""
    straddles = MatchedGap(
        RICH, POOR, poor_auc=0.80, rich_auc=0.86, difference=(-0.02, 0.01, 0.05), positives=20, negatives=180
    )
    clears = MatchedGap(
        RICH, POOR, poor_auc=0.70, rich_auc=0.86, difference=(0.03, 0.06, 0.09), positives=20, negatives=180
    )
    assert straddles.explained()
    assert not clears.explained()


def test_line() -> None:
    """The row carries both scores, the denominator, and the verdict."""
    line = MatchedGap(
        RICH, POOR, poor_auc=0.70, rich_auc=0.86, difference=(0.03, 0.06, 0.09), positives=20, negatives=180
    ).line()
    assert RICH in line
    assert POOR in line
    assert "n=20+/180-" in line
    assert "SURVIVES matching" in line


def test_report() -> None:
    """One header naming the corpus, then one row per pair."""
    predictions, truth = _corpus(0.9, 0.3)
    lines = PrevalenceMatch(resamples=DRAWS).report(predictions, truth, pairs=((RICH, POOR),))
    assert lines[0].startswith("200 studies")
    assert len(lines) == 2
    assert RICH in lines[1]
