"""A teacher's predictions written over the studies it held out, leaving every other row's labels alone."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from core.findings import STUDY, TARGETS
from kneemri.training.pseudo_table import CONFIDENCE_SUFFIX, PSEUDO_VERDICT, VERDICT_SUFFIX, PseudoTable

STUDIES = ["s1", "s2", "s3", "s4"]
TRUE_LABEL = 1.0
TRUE_CONFIDENCE = 0.9
PSEUDO_CONFIDENCE = 0.4


def _base() -> pd.DataFrame:
    """Four studies, every finding positive, every confidence high — so any write is visible."""
    columns: dict[str, list[str] | list[float]] = {STUDY: STUDIES}
    for finding in TARGETS:
        columns[finding] = [TRUE_LABEL] * len(STUDIES)
        columns[f"{finding}{CONFIDENCE_SUFFIX}"] = [TRUE_CONFIDENCE] * len(STUDIES)
        columns[f"{finding}{VERDICT_SUFFIX}"] = ["YES"] * len(STUDIES)
    return pd.DataFrame(columns)


def _predictions(studies: list[str], value: float) -> NDArray[np.float64]:
    """One flat prediction per study and finding, distinct from the true label it replaces."""
    return np.full((len(studies), len(TARGETS)), value, dtype=np.float64)


def test_covered() -> None:
    """A teacher study the base table does not carry is dropped rather than joined by position."""
    teacher = ["s3", "absent", "s1"]
    pseudo = PseudoTable(_base(), teacher, _predictions(teacher, 0.2))
    assert pseudo.covered() == ["s3", "s1"]


def test_written() -> None:
    """The covered rows take the teacher's value, its declared confidence and the `PSEUDO` verdict."""
    teacher = ["s2", "s3"]
    written = PseudoTable(_base(), teacher, _predictions(teacher, 0.2)).written(PSEUDO_CONFIDENCE)
    rows = written.set_index(STUDY)
    finding = TARGETS[0]
    assert rows.loc["s2", finding] == 0.2
    assert rows.loc["s2", f"{finding}{CONFIDENCE_SUFFIX}"] == PSEUDO_CONFIDENCE
    assert rows.loc["s2", f"{finding}{VERDICT_SUFFIX}"] == PSEUDO_VERDICT


def test_written_leaves_the_untouched_rows_true() -> None:
    """Every study the teacher did not hold out keeps its reading, its confidence and its verdict."""
    teacher = ["s2"]
    written = PseudoTable(_base(), teacher, _predictions(teacher, 0.2)).written(PSEUDO_CONFIDENCE)
    rows = written.set_index(STUDY)
    finding = TARGETS[-1]
    assert list(written[STUDY]) == STUDIES
    assert rows.loc["s1", finding] == TRUE_LABEL
    assert rows.loc["s1", f"{finding}{CONFIDENCE_SUFFIX}"] == TRUE_CONFIDENCE
    assert rows.loc["s4", f"{finding}{VERDICT_SUFFIX}"] == "YES"


def test_coverage() -> None:
    """The report states both denominators and the distance the written cells moved."""
    teacher = ["s2", "absent"]
    lines = PseudoTable(_base(), teacher, _predictions(teacher, 0.25)).coverage()
    assert "4 base studies, 2 teacher studies, 1 written" in lines[0]
    assert "1 teacher studies absent" in lines[1]
    assert "0.7500" in lines[2]
