"""The ratio that decides whether the label reader's confident errors reached the model.

The instrument's whole value is that 1 and 0 mean opposite things about where the card goes next, so the
two tests that matter are the two POLES: a member built to track the lexicon must read near 1, and a member
built to track the radiologist must read near 0, on the same cells. Everything else here guards the ways
the ratio could report a number when it should refuse.
"""

import numpy as np
import pandas as pd
import pytest

from core.data.labels.reader_agreement import GRADED, HEDGED
from core.findings import TARGETS
from kneemri.evaluation.noise_binding import (
    AGREED_NEGATIVE,
    AGREED_POSITIVE,
    MISSED,
    OVER_CALLED,
    UNREAD,
    Binding,
    NoiseBinding,
    UndiscriminatingMemberError,
)

RESAMPLES = 64
"""Enough draws for an interval to exist; this is a unit test, not the reported number."""

CONFIDENT = 0.9
UNCONVINCED = 0.1


def _cells() -> pd.DataFrame:
    """One cell of each kind, in the frame `ReaderAgreement.cells` produces."""
    rows = [
        ("s0", "en", False, False, TARGETS[0], "YES", 1.0),
        ("s0", "en", False, False, TARGETS[1], "YES", 0.0),
        ("s1", "de", False, False, TARGETS[0], "NO", 0.0),
        ("s1", "de", False, False, TARGETS[1], "NO", 1.0),
        ("s2", "en", False, False, TARGETS[0], "UNK", 0.0),
    ]
    return pd.DataFrame(rows, columns=["study", "language", GRADED, HEDGED, "target", "verdict", "truth"])


def _scored(over_called: float, missed: float = UNCONVINCED) -> pd.DataFrame:
    """Cells already carrying a member's probability, two studies deep so the bootstrap has something to draw."""
    rows = []
    for study in ("s0", "s1"):
        rows.extend(
            [
                (study, AGREED_POSITIVE, CONFIDENT),
                (study, AGREED_NEGATIVE, UNCONVINCED),
                (study, OVER_CALLED, over_called),
                (study, MISSED, missed),
            ]
        )
    return pd.DataFrame(rows, columns=["study", "group", "probability"])


def test_groups() -> None:
    """Every (verdict, truth) pair lands in its own group, and the silence is named rather than scored."""
    named = NoiseBinding().groups(_cells())

    assert list(named["group"]) == [AGREED_POSITIVE, OVER_CALLED, AGREED_NEGATIVE, MISSED, UNREAD], (
        "the four scored groups plus the abstention, in the order the cells were given"
    )


def test_probabilities() -> None:
    """A cell is joined by study NAME and finding POSITION, and a study the member never read is dropped."""
    member = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float64)

    scored = NoiseBinding().probabilities(_cells(), ["s0", "s1"], member)

    assert list(scored["study"].unique()) == ["s0", "s1"], "s2 is absent from the member's corpus and is dropped"
    assert scored.loc[scored["target"] == TARGETS[1], "probability"].tolist() == [0.22, 0.44], (
        "the second finding reads the second COLUMN, for both studies"
    )


def test_tracking() -> None:
    """The two poles of the instrument: a lexicon-tracking member reads 1, a radiologist-tracking one reads 0."""
    binding = NoiseBinding(resamples=RESAMPLES)

    lexicon = binding.tracking(_scored(over_called=CONFIDENT))
    radiologist = binding.tracking(_scored(over_called=UNCONVINCED))

    assert lexicon.ratios[OVER_CALLED] == pytest.approx(1.0), "as confident on invented positives as on real ones"
    assert radiologist.ratios[OVER_CALLED] == pytest.approx(0.0), "as unconvinced on them as on true negatives"
    assert lexicon.counts[OVER_CALLED] == 2, "the census counts cells, not studies"


def test_tracking_refuses_a_member_that_does_not_separate_the_poles() -> None:
    """A ratio over a denominator indistinguishable from zero is refused, not reported."""
    flat = _scored(over_called=CONFIDENT)
    flat["probability"] = CONFIDENT

    with pytest.raises(UndiscriminatingMemberError, match="separates the agreed poles"):
        NoiseBinding(resamples=RESAMPLES).tracking(flat)


def test_verdict() -> None:
    """The ratio is never handed back as a bare number — each end names the world it puts us in."""
    binding = Binding(
        ratios={OVER_CALLED: 1.0, MISSED: 0.0},
        intervals={OVER_CALLED: (0.8, 1.2), MISSED: (-0.2, 0.2)},
        means={AGREED_POSITIVE: CONFIDENT, AGREED_NEGATIVE: UNCONVINCED},
        counts={OVER_CALLED: 2, MISSED: 2},
    )

    assert "LEXICON" in binding.verdict(OVER_CALLED)
    assert "RADIOLOGIST" in binding.verdict(MISSED)


def test_lines() -> None:
    """The report carries the census, both ratios with intervals, and the sentence that reads them."""
    lines = NoiseBinding(resamples=RESAMPLES).tracking(_scored(over_called=CONFIDENT)).lines()

    text = "\n".join(lines)
    assert OVER_CALLED in text and MISSED in text, "both disputed groups are reported"
    assert "LIVE" in text and "CLOSED" in text, "and the two lanes the number decides between are named"
