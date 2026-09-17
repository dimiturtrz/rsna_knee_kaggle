"""The shelf's fold composition — the denominator a tail screen ranks inside."""

from pathlib import Path

import torch

from kneemri.evaluation.split_census import ShelfSplits, SplitCensus, SplitCount


def _member(models: Path, stem: str, options: dict[str, object] | None) -> None:
    """A holdout record, plus the checkpoint whose stamp the census reads — or no checkpoint at all."""
    (models / f"{stem}.holdout.npz").write_bytes(b"")
    if options is not None:
        torch.save({"options": options}, models / f"{stem}.pt")


def _stamp(fold: str) -> dict[str, object]:
    """A recipe stamp carrying the three fields the census groups on."""
    return {"fold": fold, "folds": "5", "split_seed": "2026", "variant": "small"}


def test_line() -> None:
    """A row spells the WHOLE stamp, so a fold number cannot be read across two splits."""
    assert SplitCount(fold="0", folds="5", seed="2026", members=345).line() == (" 345  fold 0 of 5, split seed 2026")


def test_lines() -> None:
    """The largest stratum's share is what says a screen over aligned members is one fold's statement."""
    rows = (
        SplitCount(fold="0", folds="5", seed="2026", members=9),
        SplitCount(fold="1", folds="5", seed="2026", members=1),
    )
    reported = "\n".join(ShelfSplits(rows=rows, unreadable=2).lines())
    assert "10 member(s) with a readable split stamp, 2 unreadable" in reported
    assert "90.0% of the shelf" in reported


def test_lines_says_so_when_nothing_is_stamped() -> None:
    """An unstamped shelf is a denominator statement and may not read as a composition."""
    assert "no member" in ShelfSplits(rows=(), unreadable=4).lines()[0]


def test_stamp(tmp_path: Path) -> None:
    """A stamped checkpoint yields its (fold, folds, seed) as strings."""
    _member(tmp_path, "fitted", _stamp("3"))
    assert SplitCensus(tmp_path).stamp("fitted") == ("3", "5", "2026")


def test_stamp_is_none_without_a_checkpoint(tmp_path: Path) -> None:
    """A holdout record whose checkpoint is gone carries no split, so it may not be counted."""
    _member(tmp_path, "lost", None)
    assert SplitCensus(tmp_path).stamp("lost") is None


def test_stamp_is_none_when_the_stamp_is_incomplete(tmp_path: Path) -> None:
    """A checkpoint missing any of the three fields cannot be grouped and counts as unreadable."""
    _member(tmp_path, "partial", {"fold": "0", "folds": "5"})
    assert SplitCensus(tmp_path).stamp("partial") is None


def test_census(tmp_path: Path) -> None:
    """The shelf groups commonest-first, and an unstamped member is reported rather than dropped."""
    _member(tmp_path, "a", _stamp("0"))
    _member(tmp_path, "b", _stamp("0"))
    _member(tmp_path, "c", _stamp("1"))
    _member(tmp_path, "d", None)
    census = SplitCensus(tmp_path).census()
    assert [(row.fold, row.members) for row in census.rows] == [("0", 2), ("1", 1)]
    assert census.unreadable == 1
