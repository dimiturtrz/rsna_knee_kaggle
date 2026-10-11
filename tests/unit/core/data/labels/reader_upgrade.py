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


def test_interval_lines_are_pinned():
    """The two printed intervals, unchanged by moving their percentile tail into `core.bootstrap`."""
    truth = {(study, cell): float((cell * study) % 3 == 0) for study in range(30) for cell in range(6)}
    lexicon = _cells([(f"s{s}", f"t{c}", ABSENT if (c * s) % 4 else POSITIVE, v) for (s, c), v in truth.items()])
    candidate = _cells([(f"s{s}", f"t{c}", POSITIVE if (c + s) % 3 else ABSENT, v) for (s, c), v in truth.items()])
    mute = _cells([(f"s{s}", f"t{c}", SILENT, 0.0) for s, c in truth])

    assert "paired interval [+0.044, +0.183]" in ReaderUpgrade(lexicon, candidate).correction_lines()[1]
    assert "blank-pair accuracy interval [0.667, 0.667] over studies" in ReaderUpgrade(mute, candidate).report()


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


def test_matched():
    """Only cells BOTH readers assert survive, and each one's verdict is carried on its own column."""
    lexicon = _cells([("a", "acl", POSITIVE, 1.0), ("a", "pcl", POSITIVE, 0.0), ("a", "mcl", SILENT, 1.0)])
    candidate = _cells([("a", "acl", ABSENT, 1.0), ("a", "pcl", SILENT, 0.0), ("a", "mcl", POSITIVE, 1.0)])

    matched = ReaderUpgrade(lexicon, candidate).matched()

    assert list(matched["target"]) == ["acl"], "a cell either reader abstained on prices no correction"
    assert list(matched["verdict_lexicon"]) == [POSITIVE]
    assert list(matched["verdict_candidate"]) == [ABSENT]


def test_correction():
    """Both rates are read on the same cells against the same truth, which is the whole point of matching."""
    matched = pd.DataFrame(
        [
            ("a", "acl", POSITIVE, 1.0, POSITIVE),
            ("a", "pcl", POSITIVE, 0.0, ABSENT),
            ("b", "acl", ABSENT, 1.0, ABSENT),
            ("b", "pcl", ABSENT, 0.0, ABSENT),
        ],
        columns=["study", "target", "verdict_lexicon", "truth", "verdict_candidate"],
    )

    shared, lexicon, candidate = ReaderUpgrade.correction(matched)

    assert shared == 4
    assert lexicon == pytest.approx(0.5), "the lexicon reads two of the four right"
    assert candidate == pytest.approx(0.75), "the candidate corrects one of the two it disagrees on"


def test_correction_reads_nan_on_a_disjoint_pair():
    """No shared cell means no correction to price, and it must not read as a tie at zero."""
    shared, lexicon, candidate = ReaderUpgrade.correction(
        pd.DataFrame(columns=["study", "target", "verdict_lexicon", "truth", "verdict_candidate"])
    )

    assert shared == 0
    assert np.isnan(lexicon) and np.isnan(candidate)


def test_resampled_correction():
    """Seeded, PAIRED, and centred on the gap it is an interval about — a tie must bracket zero."""
    rows = [
        (f"s{study}", f"t{cell}", POSITIVE, float(cell % 2), POSITIVE if cell % 2 else ABSENT)
        for study in range(30)
        for cell in range(6)
    ]
    matched = pd.DataFrame(rows, columns=["study", "target", "verdict_lexicon", "truth", "verdict_candidate"])
    upgrade = ReaderUpgrade(_cells([]), _cells([]))

    drawn = upgrade.resampled_correction(matched)

    assert np.array_equal(drawn, upgrade.resampled_correction(matched)), "same seed, same interval"
    _, lexicon, candidate = ReaderUpgrade.correction(matched)
    low, high = np.percentile(drawn, (2.5, 97.5))
    assert low <= candidate - lexicon <= high
    assert low > 0.0, "the candidate is right on every cell here and the lexicon on half of them"


def test_correction_lines():
    """The branch fires on the paired interval, not on the sign of the point estimate."""
    studies = range(30)
    lexicon = _cells([(f"s{study}", f"t{cell}", ABSENT, float(cell % 2)) for study in studies for cell in range(6)])
    corrector = _cells(
        [
            (f"s{study}", f"t{cell}", POSITIVE if cell % 2 else ABSENT, float(cell % 2))
            for study in studies
            for cell in range(6)
        ]
    )

    assert any("CORRECTS:" in line for line in ReaderUpgrade(lexicon, corrector).correction_lines())
    assert any("DOES NOT CORRECT" in line for line in ReaderUpgrade(lexicon, lexicon).correction_lines())
    assert ReaderUpgrade(lexicon, _cells([])).correction_lines() == [
        "no cell is asserted by both readers; the correction lane is unreadable here"
    ]
