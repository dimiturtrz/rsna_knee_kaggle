"""The draw weights, which decide which studies an epoch descends on.

A weight table is a pure function of a labels csv, so nothing here needs a data root — which is the
point: while the same arithmetic lived in an arm's shell line it could only be exercised by spending
the card.
"""

from pathlib import Path

import numpy as np
import pytest

from kneemri.evaluation.column_spread import BASE, ColumnSpread

HEADER = "StudyInstanceUID,MCL,MCL__verdict,Lateral OA,Lateral OA__verdict,ACL,ACL__verdict"
ROWS = (
    "a,0.82,YES,0.28,UNK,0.08,NO",
    "b,0.08,NO,0.82,YES,0.08,NO",
    "c,0.28,UNK,0.28,UNK,0.82,YES",
    "d,0.08,NO,0.08,NO,0.08,NO",
)


def _table(tmp_path: Path) -> Path:
    path = tmp_path / "labels.csv"
    path.write_text("\n".join((HEADER, *ROWS)) + "\n", encoding="utf-8")
    return path


def test_over(tmp_path: Path):
    """Positive is the VERDICT on any named column — an UNK is an unread finding, not a positive."""
    spread = ColumnSpread.over(_table(tmp_path), ("MCL", "Lateral OA"))

    assert spread.studies == ["a", "b", "c", "d"]
    assert list(spread.positive) == [True, True, False, False], "c is positive on ACL, which was not named"


def test_over_refuses_a_column_that_is_not_a_finding(tmp_path: Path):
    """A typo would otherwise raise a bare KeyError on a `__verdict` column nobody named."""
    with pytest.raises(ValueError, match="are not findings"):
        ColumnSpread.over(_table(tmp_path), ("MCL", "Lateral OAA"))


def test_spread(tmp_path: Path):
    """The contrast is against the unit, so a positive draws exactly `ratio` times a negative."""
    weights = ColumnSpread.over(_table(tmp_path), ("MCL",)).spread(6.0)

    assert list(weights) == [6.0, BASE, BASE, BASE]


def test_mass(tmp_path: Path):
    """The share of the epoch the positives buy — the number an arm pre-registers against."""
    spread = ColumnSpread.over(_table(tmp_path), ("MCL", "Lateral OA"))

    assert spread.mass(1.0) == pytest.approx(0.5), "at no contrast the mass is just the prevalence"
    assert spread.mass(6.0) == pytest.approx(12.0 / 14.0)
    assert np.isclose(spread.mass(6.0), 0.857, atol=1e-3)
