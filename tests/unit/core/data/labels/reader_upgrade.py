"""The blank-pair gate, over cell frames this test writes itself."""

import numpy as np
import pandas as pd
import pytest

from core.data.labels.reader_upgrade import ReaderUpgrade

POSITIVE = "YES"
ABSENT = "NO"
SILENT = "UNK"


def _cells(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """(study, target, verdict, truth) rows in the shape `ReaderAgreement.cells` hands back."""
    return pd.DataFrame(rows, columns=["study", "target", "verdict", "truth"])


def test_blank():
    """Only the pairs the LEXICON abstained on come through, joined on identity and not on position."""
    lexicon = _cells([("a", "acl", SILENT, 1.0), ("a", "pcl", POSITIVE, 1.0)])
    candidate = _cells([("a", "pcl", ABSENT, 1.0), ("a", "acl", POSITIVE, 1.0)])

    blank = ReaderUpgrade(lexicon, candidate).blank()

    assert list(blank["target"]) == ["acl"], "the pair the lexicon read is not headroom"
    assert list(blank["verdict"]) == [POSITIVE]


def test_answered():
    """The complement: the ground a coverage gain must not buy with a regression."""
    lexicon = _cells([("a", "acl", SILENT, 1.0), ("a", "pcl", POSITIVE, 1.0)])
    candidate = _cells([("a", "acl", POSITIVE, 1.0), ("a", "pcl", ABSENT, 1.0)])

    answered = ReaderUpgrade(lexicon, candidate).answered()

    assert list(answered["target"]) == ["pcl"]


def test_accuracy():
    """Abstentions are not scored, and the baseline is the majority class on the very same cells."""
    cells = _cells(
        [
            ("a", "acl", POSITIVE, 1.0),
            ("a", "pcl", ABSENT, 0.0),
            ("b", "acl", POSITIVE, 0.0),
            ("b", "pcl", SILENT, 1.0),
        ]
    )

    spoken, correct, majority = ReaderUpgrade.accuracy(cells)

    assert spoken == 3, "the abstention is not an assertion and cannot be right or wrong"
    assert correct == pytest.approx(2 / 3)
    assert majority == pytest.approx(2 / 3), "two of the three asserted cells are absent"


def test_accuracy_reads_nan_when_the_candidate_never_speaks():
    """An all-abstention table has no accuracy to report, and must not read as zero."""
    spoken, correct, majority = ReaderUpgrade.accuracy(_cells([("a", "acl", SILENT, 1.0)]))

    assert spoken == 0
    assert np.isnan(correct) and np.isnan(majority)


def test_resampled():
    """The interval is seeded and brackets the point estimate it is an interval about."""
    rows = [(f"s{study}", f"t{cell}", POSITIVE, float(cell % 4 > 0)) for study in range(30) for cell in range(6)]
    upgrade = ReaderUpgrade(_cells([]), _cells(rows))
    cells = _cells(rows)

    drawn = upgrade.resampled(cells)

    assert np.array_equal(drawn, upgrade.resampled(cells)), "same seed, same interval"
    low, high = np.percentile(drawn, (2.5, 97.5))
    assert low <= ReaderUpgrade.accuracy(cells)[1] <= high


def test_report():
    """A reader that beats the majority class licenses the sweep; one that abstains says so instead."""
    lexicon = _cells([(f"s{study}", f"t{cell}", SILENT, 0.0) for study in range(30) for cell in range(6)])
    truth = [(f"s{study}", f"t{cell}", float(cell % 2)) for study in range(30) for cell in range(6)]
    reads = _cells([(s, t, POSITIVE if v else ABSENT, v) for s, t, v in truth])
    mute = _cells([(s, t, SILENT, v) for s, t, v in truth])

    assert any("READS THE BLANKS" in line for line in ReaderUpgrade(lexicon, reads).report())
    assert any("SILENT TOO" in line for line in ReaderUpgrade(lexicon, mute).report())
