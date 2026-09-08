"""The join, on the failure it exists to prevent: a tag table in a different row order.

An assignment by position passes every type check and every downstream consumer, and mislabels the
corpus of most rows. So the reversed-order case is the one worth a test, not the aligned one.
"""

import pandas as pd
import pytest

from core.data.corpus.corpus_column import CorpusTags, main


def _labels() -> pd.DataFrame:
    return pd.DataFrame({"StudyInstanceUID": ["a", "b", "c"], "ACL": [0.9, 0.5, 0.1]})


def _tags() -> pd.DataFrame:
    return pd.DataFrame(
        {"StudyInstanceUID": ["c", "a", "b"], "ACL": [0.0, 0.0, 0.0], "corpus": ["rijeka", "canonical", "mrnet"]}
    )


def test_tagged():
    """The tag follows the identifier, not the row, and the finding values survive the merge."""
    joined = CorpusTags.tagged(_labels(), _tags())

    assert dict(zip(joined["StudyInstanceUID"], joined["corpus"], strict=True)) == {
        "a": "canonical",
        "b": "mrnet",
        "c": "rijeka",
    }
    assert list(joined["ACL"]) == [0.9, 0.5, 0.1]


def test_tagged_rejects_uncovered_identifier():
    """A study the tag table cannot answer is a refusal, never a NaN corpus the sampler would weight."""
    with pytest.raises(ValueError, match="not one-to-one"):
        CorpusTags.tagged(_labels(), _tags().iloc[:2])


def test_tagged_rejects_already_tagged():
    """Joining onto a table that already carries the column would duplicate it under a suffix."""
    with pytest.raises(ValueError, match="already carries"):
        CorpusTags.tagged(_tags(), _tags())


def test_tagged_rejects_untagged_tags():
    """The tag table is identified by the column it carries, not by its filename."""
    with pytest.raises(ValueError, match="no `corpus` column"):
        CorpusTags.tagged(_labels(), _labels())


def test_main(monkeypatch: pytest.MonkeyPatch):
    """The parser CONSTRUCTS, which is the step a flag collision dies in."""
    monkeypatch.setattr("sys.argv", ["corpus_column", "--help"])

    with pytest.raises(SystemExit) as exit_code:
        main()

    assert exit_code.value.code == 0
