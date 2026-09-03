"""The vision probe's pure half: montage geometry, answer parsing, and the three numbers it reports.

The server-side methods are excluded by name in the module itself — they need a model on a card, which
is an integration concern. What is tested here is everything a wrong answer could hide behind: that the
montage lays slices out in the order the prompt claims, that an unparseable answer lands on the middle
rung rather than a negative, and that each of the three reported numbers is the one it says it is.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.data.labels.image_probe import (
    FrozenSplit,
    Montage,
    ProbeCache,
    ProbeResult,
    Reading,
)
from core.data.labels.vision_prompt import GRADES
from core.findings import STUDY

SLOT_STRIDE = 20
"""Spacing between slots' fill values, so every slice in the fixture is a distinct uint8."""


def volume(slots: int = 6, slices: int = 12, size: int = 4) -> np.ndarray:
    """A cache-shaped block whose every slice is a distinct constant, so layout is checkable by value."""
    block = np.zeros((slots, slices, size, size), dtype=np.uint8)
    for slot in range(slots):
        for index in range(slices):
            block[slot, index] = slot * SLOT_STRIDE + index
    return block


def reading(study: str, grade: str, because: str = "") -> Reading:
    """One reading, without going near a model."""
    return Reading(study=study, grade=grade, because=because, raw="{}")


def test_tile() -> None:
    """Slices land left to right then down, from the requested slot only."""
    grid = Montage(slot=1, columns=4).tile(volume())
    assert grid.shape == (12, 16)
    assert grid[0, 0] == SLOT_STRIDE
    assert grid[0, 4] == SLOT_STRIDE + 1
    assert grid[4, 0] == SLOT_STRIDE + 4
    assert grid[8, 12] == SLOT_STRIDE + 11


def test_slices_of() -> None:
    """The band cuts before the layout, and the stretch takes each slice to its own full range."""
    assert [int(plane[0, 0]) for plane in Montage(slot=1, band=(4, 7)).slices_of(volume())] == [
        SLOT_STRIDE + 4,
        SLOT_STRIDE + 5,
        SLOT_STRIDE + 6,
    ]
    block = np.zeros((1, 2, 8, 8), dtype=np.uint8)
    block[0, 0] = np.arange(100, 164, dtype=np.uint8).reshape(8, 8)
    stretched = Montage(stretch=True).slices_of(block)
    assert stretched[0].min() == 0
    assert stretched[0].max() == 255
    assert stretched[1].max() == 0


def test_slices_of_rescales_a_slice_that_already_spans_the_range() -> None:
    """The whole point of the percentile window: a 0-255 slice is still re-windowed, not left alone.

    Min-max would be a no-op on the cache, which stores every slice at full range already — it was
    run as one, and three models answering identically read as a result rather than as a null.
    """
    block = np.zeros((1, 1, 10, 10), dtype=np.uint8)
    block[0, 0] = 40
    block[0, 0, 0, 0] = 0
    block[0, 0, 9, 9] = 255
    assert block[0, 0].min() == 0
    assert block[0, 0].max() == 255
    assert not np.array_equal(Montage(stretch=True).slices_of(block)[0], block[0, 0])


def test_tile_pads_a_ragged_grid() -> None:
    """A slice count that does not divide the column count is padded, never silently truncated."""
    grid = Montage(columns=5).tile(volume(slices=12))
    assert grid.shape == (12, 20)
    assert grid[8, 10] == 0


def test_parse() -> None:
    """A well-formed answer yields its grade and reason, even with prose wrapped around the JSON."""
    parsed = Reading.parse("s1", 'Here you go: {"acl": "intact", "because": "continuous fibres"} done')
    assert parsed.grade == "intact"
    assert parsed.because == "continuous fibres"


def test_parse_falls_to_the_middle_rung() -> None:
    """Unparseable or out-of-vocabulary answers become `possible`, never a negative call."""
    assert Reading.parse("s1", "no json at all").grade == "possible"
    assert Reading.parse("s2", json.dumps({"acl": "shredded"})).grade == "possible"


def test_score() -> None:
    """The three rungs map onto the scale AUC is computed over, in order."""
    assert [reading("s", grade).score for grade in GRADES] == [0.0, 0.5, 1.0]


def test_matrix() -> None:
    """Only `torn` is a positive call; `possible` sits on the negative side of the cut."""
    result = ProbeResult((reading("a", "torn"), reading("b", "torn"), reading("c", "possible")), (1, 0, 1))
    assert result.matrix() == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 0,
    }


def test_call_rate() -> None:
    """How often the model said torn — the number that exposes a constant answer."""
    assert ProbeResult((reading("a", "torn"), reading("b", "intact")), (1, 0)).call_rate() == 0.5


def test_base_rate() -> None:
    """How often the truth is positive."""
    assert ProbeResult((reading("a", "torn"), reading("b", "torn")), (1, 0)).base_rate() == 0.5


def test_auc() -> None:
    """Perfect separation reads 1.0, and a constant answer reads 0.5 because every pair ties."""
    perfect = ProbeResult((reading("a", "torn"), reading("b", "intact")), (1, 0))
    assert perfect.auc() == 1.0
    constant = ProbeResult((reading("a", "torn"), reading("b", "torn")), (1, 0))
    assert constant.auc() == 0.5


def test_degenerate_rate() -> None:
    """Identical prose across studies counts; genuinely different prose does not."""
    same = (reading("a", "torn", "the ligament is disrupted"), reading("b", "torn", "the ligament is disrupted"))
    assert ProbeResult(same, (1, 0)).degenerate_rate() == 1.0
    apart = (reading("a", "torn", "fibres are disrupted"), reading("b", "intact", "a taut dark band throughout"))
    assert ProbeResult(apart, (1, 0)).degenerate_rate() == 0.0


def test_frame() -> None:
    """Every reading becomes a row carrying its grade, score and label, so the run can be re-cut."""
    frame = ProbeResult((reading("a", "torn", "why"),), (1,)).frame()
    assert list(frame.columns) == [STUDY, "grade", "score", "acl", "because", "raw"]
    assert frame.iloc[0]["score"] == 1.0
    assert frame.iloc[0]["acl"] == 1


def test_report() -> None:
    """The three numbers are all present, and the base rate is printed beside the call rate."""
    spoken = ProbeResult((reading("a", "torn"), reading("b", "intact")), (1, 0)).report()
    assert "base_rate=0.500" in spoken
    assert "call_rate=0.500" in spoken
    assert "auc=1.000" in spoken
    assert "degenerate_pair_rate" in spoken


def test_positions(tmp_path: Path) -> None:
    """Each study maps to its own row of the cache, in the order the index file lists them."""
    index = tmp_path / "index.json"
    index.write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")
    assert ProbeCache(images=tmp_path / "unread.npy", index=index).positions() == {"a": 0, "b": 1, "c": 2}


def test_volumes(tmp_path: Path) -> None:
    """The cache comes back memmapped, never read whole — it is 35 GB on the real run."""
    images = tmp_path / "images.npy"
    np.save(images, volume()[None])
    loaded = ProbeCache(images=images, index=tmp_path / "unread.json").volumes()
    assert isinstance(loaded, np.memmap)
    assert loaded.shape == (1, 6, 12, 4, 4)


def test_tune(tmp_path: Path) -> None:
    """Only the tune half is returned; the holdout half is never handed to a caller."""
    split = tmp_path / "split.csv"
    pd.DataFrame({STUDY: ["a", "b", "c"], "split": ["tune", "holdout", "tune"]}).to_csv(split, index=False)
    assert FrozenSplit(split).tune() == ["a", "c"]
