"""What the shipped traversal opened, counted off the memo rather than off the arithmetic."""

from pathlib import Path

import torch

from core.data.corpus.decode_bench import CHAMPION_GROUPS
from kneemri.kaggle.traversal_census import CensusRow, TraversalCensus


def _bag(value: float) -> list[dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    return [{336: (torch.full((1, 2, 2, 2), value), torch.ones(1))}]


def test_reads():
    """One `Read` per champion group, each carrying that group's band and depth."""
    reads = TraversalCensus.reads()

    assert [(read.slices, (read.reading.band.low, read.reading.band.high)) for read in reads] == list(CHAMPION_GROUPS)


def test_equal():
    """Byte equality, and a mask that differs is a difference even when every pixel matches."""
    assert TraversalCensus.equal(_bag(1.0), _bag(1.0))
    assert not TraversalCensus.equal(_bag(1.0), _bag(2.0))
    assert not TraversalCensus.equal(_bag(1.0), []), "a missing read is not an equal read"

    masked = _bag(1.0)
    masked[0][336] = (masked[0][336][0], torch.zeros(1))
    assert not TraversalCensus.equal(_bag(1.0), masked), "the presence mask counts too"


def test_census(tmp_path: Path):
    """An unreadable study costs no opens under either traversal, and the two still agree.

    The failure path rather than a corpus because that is what a unit may touch — the real bills come
    from the CLI over the split, which is the only place the number means anything.
    """
    row = TraversalCensus.census(tmp_path, {})

    assert (row.slices, row.slices_apart, row.headers, row.headers_apart) == (0, 0, 0, 0)
    assert row.matched, "zeros under one memo are the zeros under three"


def test_report():
    """Both bills per study and the share the memo removed, plus how many studies agreed."""
    lines = TraversalCensus.report([CensusRow(slices=4, slices_apart=8, headers=1, headers_apart=3, matched=True)])

    assert "chunk-outer    4.0" in lines[1] and "removed 50.0%" in lines[1]
    assert "byte-identical bags: 1 of 1 studies" in lines[3]
    assert TraversalCensus.report([])[0].startswith("0 studies"), "an empty census divides by one, not by zero"
