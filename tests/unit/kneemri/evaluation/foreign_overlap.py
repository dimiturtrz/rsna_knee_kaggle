"""The contamination guard, on tables written by hand so the overlap is known before it is measured."""

from pathlib import Path

import pandas as pd

from kneemri.evaluation.foreign_overlap import IDENTIFIER, ForeignOverlap

STUDIES = 40
"""Enough that a five-fold hash assignment puts a workable number of studies in fold 0."""


def _table(path: Path, names: list[str]) -> Path:
    """A table carrying only the join key, which is all this tool ever reads from either side."""
    pd.DataFrame({IDENTIFIER: names, "Report": [f"report for {name}" for name in names]}).to_csv(path, index=False)
    return path


def test_holdout(tmp_path: Path):
    """One fold's studies, and the five folds partition the corpus rather than overlapping or dropping."""
    names = [f"s{index}" for index in range(STUDIES)]
    train = _table(tmp_path / "train.csv", names)
    overlap = ForeignOverlap()

    folds = [overlap.holdout(train, fold) for fold in range(5)]

    assert sorted(name for fold in folds for name in fold) == sorted(names), "every study lands in exactly one fold"
    assert all(fold for fold in folds), "and no fold is empty at this corpus size"


def test_covered(tmp_path: Path):
    """The intersection, in holdout order, with a foreign table that carries extra rows and misses some.

    Three equivalence classes in one call: a study they carry, a study they lack, and a row of theirs
    that is not ours at all — the last must not inflate the count, since their corpus may be larger.
    """
    holdout = ["s3", "s1", "s7"]
    foreign = _table(tmp_path / "theirs.csv", ["s7", "stranger", "s3"])

    assert ForeignOverlap().covered(holdout, foreign) == ["s3", "s7"], "holdout order, and their extras ignored"


def test_covered_disjoint(tmp_path: Path):
    """A foreign table sharing no study reads as empty rather than as an error, which is the clean case."""
    foreign = _table(tmp_path / "theirs.csv", ["a", "b"])

    assert ForeignOverlap().covered(["s1", "s2"], foreign) == []


def test_verdict():
    """Three readings, and the boundary between them is any overlap at all rather than a share."""
    overlap = ForeignOverlap()

    assert "CLEAN" in overlap.verdict(0, 909)[0]

    partial = overlap.verdict(896, 909)
    assert "CONTAMINATED" in partial[0], "the case that actually occurred"
    assert "98.6%" in partial[0], "the share is stated, never left to the reader"
    assert any("void" in line for line in partial), "and it says what the number may no longer be used for"

    assert "CONTAMINATED" in overlap.verdict(1, 909)[0], "one study is contamination; there is no tolerance band"
