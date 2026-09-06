"""The one guard that makes the fill a supervision fix rather than a second reader: it only writes silence."""

import pandas as pd

from core.data.labels.reader_agreement import SILENT, VERDICT_SUFFIX
from core.data.labels.table_imputation import IMPUTED, SilenceImputation

TARGET = "Synovitis"
SOURCE = "Effusion"


def _table() -> pd.DataFrame:
    """Four studies, one per equivalence class of the target's verdict: silent, stated yes, stated no, silent."""
    return pd.DataFrame(
        {
            "StudyInstanceUID": ["s0", "s1", "s2", "s3"],
            TARGET: [0.21, 0.95, 0.05, 0.21],
            f"{TARGET}{VERDICT_SUFFIX}": [SILENT, "YES", "NO", SILENT],
            SOURCE: [0.88, 0.10, 0.90, 0.04],
        }
    )


def test_undecided():
    """Silence is read off the VERDICT column, and a table without one offers no cells rather than guessing."""
    imputation = SilenceImputation(_table())

    assert imputation.undecided(TARGET).tolist() == [True, False, False, True]
    assert not SilenceImputation(_table().drop(columns=[f"{TARGET}{VERDICT_SUFFIX}"])).undecided(TARGET).any()


def test_imputed():
    """The silent cells take the source's value; a stated cell is untouched even when the source disagrees."""
    filled = SilenceImputation(_table()).imputed(TARGET, SOURCE)

    assert filled[TARGET].tolist() == [0.88, 0.95, 0.05, 0.04], "s2 keeps its stated 0.05 against a 0.90 source"
    assert filled[f"{TARGET}{VERDICT_SUFFIX}"].tolist() == [IMPUTED, "YES", "NO", IMPUTED]
    assert filled[SOURCE].tolist() == _table()[SOURCE].tolist(), "the source column is read, never written"


def test_census():
    """The count and both means are printed, because a fill of two cells and of four thousand share a name."""
    lines = SilenceImputation(_table()).census(TARGET, SOURCE)

    assert "2 of 4 cells silent, 2 stated and untouched" in lines[0]
    assert "0.2100 -> 0.4600" in lines[1], "the mean over the filled cells only, before and after"
    assert lines[2].endswith("0 -> 1"), "one of the two silent cells crosses the cut, the other does not"
    assert "0.3550 -> 0.4800" in lines[3], "and the whole column's mean, which is what the ruler sees"
