"""Readers whose splits are known by construction: one finding disputed, one read the same by everyone."""

import pandas as pd

from core.data.labels.reader_disagreement import SPLIT, ReaderDisagreement
from core.data.labels.table_blend import LabelBlend
from core.findings import STUDY, TARGETS

STUDIES = 8
DISPUTED = TARGETS[0]
"""The finding the second reader reads inverted, so every covered cell sits a full unit apart."""

AGREED = TARGETS[1]
"""The finding both readers read identically, so its spread is exactly zero on every cell."""

LONE = TARGETS[2]
"""The finding only the base carries, so no cell reaches `MIN_READERS` and the column is empty."""


def _base() -> pd.DataFrame:
    """A base table carrying all three findings, alternating so no column is degenerate."""
    return pd.DataFrame(
        [
            {STUDY: f"s{index}"} | {DISPUTED: float(index % 2), AGREED: float(index % 2), LONE: float(index % 2)}
            for index in range(STUDIES)
        ]
    )


def _foreign() -> pd.DataFrame:
    """A second reader that inverts one finding, copies another, and never saw the third."""
    return pd.DataFrame(
        [{STUDY: f"s{index}"} | {DISPUTED: float(1 - index % 2), AGREED: float(index % 2)} for index in range(STUDIES)]
    )


def _disagreement() -> ReaderDisagreement:
    """The two readers above, aligned the way the CLI aligns them."""
    return ReaderDisagreement(LabelBlend(_base(), [_foreign()]))


def test_spread():
    """An inverted reading is a full unit apart; an identical one is zero; an unread one is dropped."""
    disagreement = _disagreement()

    assert disagreement.spread(DISPUTED).tolist() == [1.0] * STUDIES
    assert disagreement.spread(AGREED).tolist() == [0.0] * STUDIES
    assert disagreement.spread(LONE).empty, "one reader cannot disagree, so the cell is unmeasured"


def test_profile():
    """Findings come back most-disputed first, with the split rate following the threshold."""
    profile = _disagreement().profile().set_index("finding")

    assert profile.index[0] == DISPUTED
    assert profile.loc[DISPUTED, "split_rate"] == 1.0, f"a full unit clears {SPLIT} on every cell"
    assert profile.loc[AGREED, "split_rate"] == 0.0
    assert profile.loc[LONE, "studies"] == 0


def test_cells():
    """The per-cell table joins by study name and carries one column per finding."""
    cells = _disagreement().cells()

    assert list(cells[STUDY]) == [f"s{index}" for index in range(STUDIES)]
    assert cells[DISPUTED].tolist() == [1.0] * STUDIES
    assert cells[LONE].isna().all(), "an unmeasured finding stays missing rather than reading as agreement"


def test_report():
    """Every finding gets a line, and the across-finding span is stated as its own line."""
    lines = _disagreement().report()

    assert any(line.startswith(DISPUTED) and "mean 1.0000" in line for line in lines)
    assert any("spread across findings 1.0000" in line for line in lines)
