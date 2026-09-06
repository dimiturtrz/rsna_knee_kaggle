import hashlib

import numpy as np
import pandas as pd

from core.findings import STUDY
from kneemri.evaluation.foreign_split import FOLDS, REPORT, ForeignSplit


def _expected(text: str) -> int:
    """Their hash computed independently of the class under test, so the test is not the code again."""
    return int(hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:8], 16) % FOLDS


def _labels(rows: dict[str, str | float]) -> pd.DataFrame:
    return pd.DataFrame({STUDY: list(rows), REPORT: list(rows.values())})


def test_group() -> None:
    """The transcription is theirs: md5 of the text, first eight hex digits as an int, mod five."""
    assert ForeignSplit._group("acl tear, medial meniscus intact") == _expected("acl tear, medial meniscus intact")
    assert 0 <= ForeignSplit._group("") < FOLDS


def test_of() -> None:
    """A study is placed by its REPORT, so two studies sharing a report land in the same fold."""
    labels = _labels({"a": "same report", "b": "same report", "c": "a different one"})
    split = ForeignSplit.of(["a", "b", "c"], labels)
    assert split.fold[0] == split.fold[1] == _expected("same report")
    assert split.fold[2] == _expected("a different one")


def test_of_absent_and_blank_are_different_cases() -> None:
    """Absent hashes the study UID; present-but-blank hashes the empty string. Collapsing them moves folds.

    Their `.fillna('')` runs before their `.get(s, s)`, so a blank report is a real value and never reaches
    the UID fallback. A port that treats blank as absent partitions the blank-report studies differently
    from the checkpoint that was actually trained.
    """
    split = ForeignSplit.of(["blank", "absent"], _labels({"blank": np.nan}))
    assert split.fold[0] == _expected(""), "blank hashes the empty string, as their fillna leaves it"
    assert split.fold[1] == _expected("absent"), "absent falls back to the study UID"


def test_of_duplicate_rows() -> None:
    """A study listed twice in the label table is one study, not an ambiguous index lookup."""
    labels = pd.DataFrame({STUDY: ["a", "a"], REPORT: ["first", "second"]})
    assert ForeignSplit.of(["a"], labels).fold.tolist() == [_expected("first")]


def test_out_of_sample() -> None:
    """Fold f held out exactly `grp == f`; everything else it trained on and may never be scored on."""
    split = ForeignSplit(fold=np.array([0, 1, 0, 2]), studies=["a", "b", "c", "d"])
    assert split.out_of_sample(0).tolist() == [True, False, True, False]
    assert split.out_of_sample(1).tolist() == [False, True, False, False]


def test_sizes() -> None:
    """Every fold is reported, an empty one included — a missing key would hide a bad join."""
    sizes = ForeignSplit(fold=np.array([0, 0, 3]), studies=["a", "b", "c"]).sizes()
    assert sizes == {0: 2, 1: 0, 2: 0, 3: 1, 4: 0}


def test_lines() -> None:
    """The reading carries BOTH denominators, because the in-sample share is the whole point."""
    lines = ForeignSplit(fold=np.array([0, 1, 1, 1]), studies=list("abcd")).lines(0)
    assert "out-of-sample on 1 of 4 studies (25.0%)" in lines[0]
    assert "IN-SAMPLE on 3 (75.0%)" in lines[0]
    assert "raptor_solo" in lines[2]
