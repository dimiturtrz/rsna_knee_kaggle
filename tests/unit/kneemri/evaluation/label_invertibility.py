"""How much of the extracted target a report embedding recovers by itself, and which band that lands in."""

from pathlib import Path

import numpy as np
import pandas as pd

from kneemri.evaluation.label_invertibility import ColumnProbe, Invertibility, LabelInvertibility

STUDIES = ("a", "b", "c", "d", "e", "f")

GRADED = (0.94, 0.08, 0.82, 0.28, 0.94, 0.08)
"""Three studies above the 0.75 recut and three below, on the five rungs the real table uses. Both the
training rows and the held-out rows carry both classes, which is what a probe and an AUC each require."""


def _corpus(tmp_path: Path) -> LabelInvertibility:
    """Six studies whose first embedding dimension IS the target, so the probe must read it perfectly."""
    np.savez(
        tmp_path / "vectors.npz",
        identifiers=np.array(STUDIES),
        vectors=np.array([[1.0 if value > 0.75 else 0.0, 0.0] for value in GRADED], dtype=np.float32),
    )
    pd.DataFrame(
        {
            "StudyInstanceUID": list(STUDIES),
            "ACL": list(GRADED),
            "ACL__conf": [0.5] * len(STUDIES),
        }
    ).to_csv(tmp_path / "labels.csv", index=False)
    np.savez(tmp_path / "held.holdout.npz", studies=np.array(["e", "f"]))
    return LabelInvertibility(tmp_path / "vectors.npz", tmp_path / "labels.csv", tmp_path)


def test_line() -> None:
    """A row carries the positive count, so a rare finding cannot hide behind an AUC."""
    assert ColumnProbe(label="ACL", area=0.8480, positives=153).line() == "0.8480   153 pos  ACL"


def test_mean() -> None:
    """The unweighted mean over findings is the ruler the board scores on."""
    rows = (ColumnProbe("ACL", 0.8, 10), ColumnProbe("MCL", 0.9, 10))
    assert Invertibility(rows=rows, studies=909).mean() == 0.8500000000000001


def test_verdict_calls_an_invertible_target_a_shortcut() -> None:
    """At or above what the best image member reads, the text branch is a copy of the label."""
    reading = Invertibility(rows=(ColumnProbe("ACL", 0.97, 10),), studies=909)
    assert reading.verdict().startswith("SHORTCUT")


def test_verdict_calls_a_weak_read_complementary() -> None:
    """Inside the band this project declines to seat an image member, the text carries its own signal."""
    reading = Invertibility(rows=(ColumnProbe("ACL", 0.55, 10),), studies=909)
    assert reading.verdict().startswith("COMPLEMENTARY")


def test_verdict() -> None:
    """The measured 0.8553 is a PARTIAL shortcut, which licenses only a staged and ablated arm."""
    reading = Invertibility(rows=(ColumnProbe("ACL", 0.8553, 10),), studies=909)
    assert reading.verdict().startswith("CONTESTED")
    assert "stage" in reading.verdict()


def test_lines() -> None:
    """The reading spells its denominator, every finding, the mean and the verdict."""
    reported = Invertibility(rows=(ColumnProbe("ACL", 0.8553, 153),), studies=909).lines()
    assert "1 finding(s) probed on 909 held-out studies" in reported[0]
    assert "mean 0.8553" in reported[-2]


def test_lines_says_so_when_nothing_could_be_probed() -> None:
    """No shared study is a denominator statement and may not read as a complementary verdict."""
    assert "no finding" in Invertibility(rows=(), studies=0).lines()[0]


def test_cut(tmp_path: Path) -> None:
    """Ten findings cut at 0.75 and the two that lose their ruler there keep the midpoint."""
    corpus = _corpus(tmp_path)
    assert corpus.cut("ACL") == 0.75
    assert corpus.cut("Baker's") == 0.5


def test_embedded(tmp_path: Path) -> None:
    """The vectors come back with the studies they belong to, in the file's own order."""
    identifiers, vectors = _corpus(tmp_path).embedded()
    assert identifiers == list(STUDIES)
    assert vectors.shape == (len(STUDIES), 2)


def test_held(tmp_path: Path) -> None:
    """The held-out studies are taken from the member's record rather than re-derived."""
    assert _corpus(tmp_path).held("held") == {"e", "f"}


def test_findings(tmp_path: Path) -> None:
    """A finding column is the one carrying no extractor metadata suffix."""
    corpus = _corpus(tmp_path)
    assert corpus.findings(pd.read_csv(corpus.labels)) == ["ACL"]


def test_area(tmp_path: Path) -> None:
    """A perfect ranking reads 1.0 and a reversed one reads 0.0, off the rank statistic alone."""
    corpus = _corpus(tmp_path)
    truth = np.array([False, False, True, True])
    assert corpus.area(np.array([0.1, 0.2, 0.8, 0.9]), truth) == 1.0
    assert corpus.area(np.array([0.9, 0.8, 0.2, 0.1]), truth) == 0.0


def test_area_is_undefined_for_a_single_class(tmp_path: Path) -> None:
    """A finding with no negative in the holdout has no rank statistic and may not report a number."""
    scores = np.array([0.1, 0.2])
    assert np.isnan(_corpus(tmp_path).area(scores, np.array([True, True])))


def test_probe(tmp_path: Path) -> None:
    """A target written into the embedding is recovered exactly, on the studies the member held out."""
    reading = _corpus(tmp_path).probe("held")
    assert reading.studies == 2
    assert [row.label for row in reading.rows] == ["ACL"]
    assert reading.rows[0].positives == 1
    assert reading.rows[0].area == 1.0
