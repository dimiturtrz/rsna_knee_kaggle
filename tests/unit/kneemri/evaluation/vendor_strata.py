from pathlib import Path

import numpy as np
import pandas as pd

from kneemri.evaluation.vendor_strata import Stratum, VendorStrata

STUDIES = 400


def _studies() -> list[str]:
    return [f"study{index:04d}" for index in range(STUDIES)]


def _table(tmp_path: Path) -> Path:
    """A metadata table shaped like the real one: half Siemens at 1.5 T, half Philips at 3 T, one absent."""
    destination = tmp_path / "canonical_metadata.csv"
    names = _studies()
    frame = pd.DataFrame(
        {
            "StudyInstanceUID": names,
            "scanner": ["Siemens"] * (STUDIES // 2) + ["Philips"] * (STUDIES // 2 - 1) + [None],
            "field_strength_t": [1.5] * (STUDIES // 2) + [3.0] * (STUDIES // 2 - 1) + [None],
        }
    )
    frame.to_csv(destination, index=False)
    return destination


def test_size():
    """The denominator every number in a row is read over, as an int rather than a numpy scalar."""
    assert Stratum("s", np.arange(7, dtype=np.int64)).size == 7
    assert Stratum("empty", np.array([], dtype=np.int64)).size == 0


def test_read(tmp_path: Path):
    """The table is indexed by StudyInstanceUID — the column the corpus writer actually emits.

    Named here rather than assumed: an `identifier` spelling raised a KeyError on first run, and a
    reader that silently produced no stratum would have reported this axis as flat.
    """
    strata = VendorStrata.read(_table(tmp_path))

    assert strata.table.index.name == "StudyInstanceUID"
    assert strata.table.loc["study0000", "scanner"] == "Siemens"


def test_scanners(tmp_path: Path):
    """The manufacturer column in the order asked for, absent where the study joins nothing."""
    column = VendorStrata.read(_table(tmp_path)).scanners(["study0000", "study0399", "absent"])

    assert column.iloc[0] == "Siemens"
    assert bool(column.isna().iloc[1]), "the table row with no scanner stays absent"
    assert bool(column.isna().iloc[2]), "a study the table does not know reindexes to absent, it does not raise"


def test_fields(tmp_path: Path):
    """The field-strength column as floats, which is what the threshold needs."""
    column = VendorStrata.read(_table(tmp_path)).fields(["study0000", "study0399"])

    assert column.iloc[0] == 1.5
    assert bool(column.isna().iloc[1])


def test_strata(tmp_path: Path):
    """Every vendor present, plus both field bands — and a study joining nothing falls in no stratum."""
    found = {stratum.name: stratum for stratum in VendorStrata.read(_table(tmp_path)).strata(_studies())}

    assert set(found) == {"Siemens", "Philips", "under 2 T", "2 T and above"}
    assert found["Siemens"].size == STUDIES // 2
    assert found["Philips"].size == STUDIES // 2 - 1, "the unjoined study is absent, not folded into a vendor"
    assert found["under 2 T"].size == STUDIES // 2
    assert list(found["Siemens"].studies[:3]) == [0, 1, 2], "strata are POSITIONS into the studies passed in"
