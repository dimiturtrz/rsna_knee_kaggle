"""The swap gate over a frame built in memory: what the gate asserts, not what a real member scores.

Every record here is synthetic and seeded. The module's own reading is a statement about the champion's
five parents; what these tests hold is the arithmetic that turns six holdout records into that
statement — which rows survive the gold drop, which truth a child is asserted against, that the guard is
the child against the REST of the block rather than against its parent, and that both halves of the gate
have to pass before a seat reads GO.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.seat_swap import (
    ANCHOR,
    SeatFrame,
    SeatSwap,
    SwapReading,
)

SEED = 2026
STUDIES = 40
PARENTS = ("p0", "p1", "p2", "p3", "p4")
GOLD_ROWS = 3


def _truth(generator: np.random.Generator) -> np.ndarray:
    """Graded truth whose first rows are radiologist overrides — exactly 0.0 or 1.0 on all twelve."""
    graded = generator.choice([0.0, 0.28, 0.5, 0.68, 0.82, 1.0], size=(STUDIES, len(TARGETS)))
    graded[:GOLD_ROWS] = (generator.random((GOLD_ROWS, len(TARGETS))) > 0.5).astype(np.float64)
    return graded.astype(np.float32)


def _holdouts(tmp_path: Path, *, child_truth_moves: bool = False) -> Path:
    """Five parents, an anchor and two children, all scored against one label table."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(SEED)
    truth = _truth(generator)
    signal = (truth > 0.5).astype(np.float64)
    studies = np.asarray([f"s{index:03d}" for index in range(STUDIES)])
    for name in (*PARENTS, ANCHOR, "child", "copycat"):
        noise = 0.05 if name == "copycat" else 0.6
        served = signal + generator.normal(0.0, noise, signal.shape)
        moved = truth + 0.01 if child_truth_moves and name == "child" else truth
        np.savez(
            tmp_path / f"{name}.holdout.npz",
            studies=studies,
            predictions=served,
            truth=moved,
            epoch=np.asarray(7),
        )
    return tmp_path


def _frame(tmp_path: Path) -> SeatFrame:
    """The frame those records make, under the published rank transform."""
    return SeatFrame.over(_holdouts(tmp_path), PARENTS, ANCHOR)


def _reading() -> SwapReading:
    """A reading with both halves of the gate satisfied, for the verdict arithmetic."""
    return SwapReading(
        child="child",
        parent="p0",
        solo_child=0.90,
        solo_parent=0.89,
        swap_gain=0.0020,
        swap_five=0.0030,
        child_rest_corr=0.90,
        parent_rest_corr=0.91,
        block_pairwise=0.87,
        parents_pairwise=0.88,
    )


def test_ranked() -> None:
    """Per-column midranks over the row count — ties share a rank rather than taking array order."""
    matrix = SeatFrame.ranked(np.asarray([[0.1, 5.0], [0.3, 5.0], [0.2, 1.0]]))
    assert matrix[:, 0].tolist() == [1 / 3, 1.0, 2 / 3]
    assert matrix[0, 1] == matrix[1, 1], "tied predictions must not be ordered by their position"


def test_flat_correlation() -> None:
    """Pearson over every cell of two rank matrices, as one number."""
    left = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    assert SeatFrame.flat_correlation(left, left) == pytest.approx(1.0)
    assert SeatFrame.flat_correlation(left, -left) == pytest.approx(-1.0)


def test_parsed_pairs() -> None:
    """`child:parent`, refused rather than guessed at when the chair is not named."""
    assert SeatSwap.parsed_pairs(["a:b", "c:d"]) == [("a", "b"), ("c", "d")]
    with pytest.raises(ValueError, match="child.*parent"):
        SeatSwap.parsed_pairs(["a"])


def test_over(tmp_path: Path) -> None:
    """The shared frame minus the gold rows, with one rank matrix per member."""
    frame = _frame(tmp_path)
    assert len(frame.studies) == STUDIES - GOLD_ROWS
    assert frame.truth.shape == (STUDIES - GOLD_ROWS, len(TARGETS))
    assert set(frame.ranks) == {*PARENTS, ANCHOR}
    assert not np.all(np.isin(frame.truth, [0.0, 1.0]), axis=1).any(), "a gold row survived the drop"


def test_record(tmp_path: Path) -> None:
    """One member's reading, and a refusal naming the member when it left none."""
    reading = SeatFrame.record(_holdouts(tmp_path), "p0")
    assert len(reading.studies) == STUDIES
    with pytest.raises(FileNotFoundError, match="ghost"):
        SeatFrame.record(tmp_path, "ghost")


def test_taken(tmp_path: Path) -> None:
    """Values joined by study NAME, and a refusal when the frame asks for a study the record lacks."""
    reading = SeatFrame.record(_holdouts(tmp_path), "p0")
    taken = SeatFrame.taken(reading, ["s002", "s000"], reading.predictions)
    assert taken[0].tolist() == reading.predictions[2].tolist()
    with pytest.raises(ValueError, match="absent"):
        SeatFrame.taken(reading, ["s999"], reading.predictions)


def test_child(tmp_path: Path) -> None:
    """A candidate's ranks over the frame — and a refusal when its truth is a different table."""
    frame = _frame(tmp_path)
    seated = frame.child("child")
    assert seated.shape == (len(frame.studies), len(TARGETS))
    moved = SeatFrame.over(_holdouts(tmp_path / "moved", child_truth_moves=True), PARENTS, ANCHOR)
    with pytest.raises(ValueError, match="different label table"):
        moved.child("child")


def test_proxy(tmp_path: Path) -> None:
    """The champion mix: the ours-side at 2/3, the anchor at 1/3 — and the ours-side alone."""
    frame = _frame(tmp_path)
    block = [frame.ranks[name] for name in PARENTS]
    mean = np.mean(block, axis=0)
    assert frame.proxy(block, with_anchor=False) == pytest.approx(mean)
    assert frame.proxy(block) == pytest.approx((2 / 3) * mean + (1 / 3) * frame.ranks[ANCHOR])


def test_macro(tmp_path: Path) -> None:
    """LEDGER75 over the frame, and a member that ranks the truth perfectly reads 1.0."""
    frame = _frame(tmp_path)
    assert 0.0 <= frame.macro(frame.ranks["p0"]) <= 1.0
    assert frame.macro(frame.truth.astype(np.float64)) == pytest.approx(1.0)


def test_block(tmp_path: Path) -> None:
    """The five chairs, with a child in exactly the one it stands in."""
    frame = _frame(tmp_path)
    seated = frame.child("child")
    block = frame.block(PARENTS, seated, "p2")
    assert block[2] is seated
    assert [one is frame.ranks[name] for one, name in zip(block, PARENTS, strict=True)] == [
        True,
        True,
        False,
        True,
        True,
    ]


def test_mean_pairwise(tmp_path: Path) -> None:
    """Mean upper-triangle corr — 1.0 for a block of one repeated member."""
    frame = _frame(tmp_path)
    assert frame.mean_pairwise([frame.ranks["p0"], frame.ranks["p0"]]) == pytest.approx(1.0)
    assert frame.mean_pairwise([frame.ranks[name] for name in PARENTS]) < 1.0


def test_guarded() -> None:
    """D43's sentence: the child may be a little closer to the rest than its parent, not a lot."""
    assert _reading().guarded()
    tight = replace(_reading(), child_rest_corr=0.95)
    assert not tight.guarded(), "a child that collapses the block's spread must not pass the guard"


def test_passes() -> None:
    """GO needs BOTH halves — a big gain with a collapsed block is still a refusal."""
    assert _reading().passes()
    assert not replace(_reading(), swap_gain=0.0005).passes()
    assert not replace(_reading(), child_rest_corr=0.99).passes()


def test_verdict() -> None:
    """Which half refused, never a bare NO-GO."""
    assert _reading().verdict() == "GO"
    assert replace(_reading(), swap_gain=0.0).verdict() == "NO-GO gain"
    assert replace(_reading(), child_rest_corr=0.99).verdict() == "NO-GO corr"


def test_line() -> None:
    """A row carries both members, the signed gain and the verdict."""
    line = _reading().line()
    assert "child" in line
    assert "+0.0020" in line
    assert line.endswith("GO")


def test_baseline(tmp_path: Path) -> None:
    """The untreated proxy and the ours-side alone, recomputed rather than cached."""
    frame = _frame(tmp_path)
    proxy, five = SeatSwap(frame, PARENTS).baseline()
    block = [frame.ranks[name] for name in PARENTS]
    assert proxy == pytest.approx(frame.macro(frame.proxy(block)))
    assert five == pytest.approx(frame.macro(frame.proxy(block, with_anchor=False)))


def test_rest(tmp_path: Path) -> None:
    """The four parents a swap does not touch — the parent it replaces is not among them."""
    frame = _frame(tmp_path)
    rest = SeatSwap(frame, PARENTS).rest("p1")
    assert rest == pytest.approx(np.mean([frame.ranks[name] for name in PARENTS if name != "p1"], axis=0))


def test_reading(tmp_path: Path) -> None:
    """One swap priced on both halves, and a refusal when the named chair is not a seat."""
    gate = SeatSwap(_frame(tmp_path), PARENTS)
    reading = gate.reading("copycat", "p0")
    assert reading.parent == "p0"
    assert reading.child_rest_corr > reading.parent_rest_corr, (
        "a member built as a near-copy of the signal has to read as LESS diverse than the parent it replaces"
    )
    assert not reading.guarded()
    with pytest.raises(ValueError, match="not one of the block"):
        gate.reading("child", "nobody")


def test_table(tmp_path: Path) -> None:
    """Every requested swap, in the order asked."""
    readings = SeatSwap(_frame(tmp_path), PARENTS).table([("child", "p0"), ("copycat", "p3")])
    assert [one.parent for one in readings] == ["p0", "p3"]


def test_lines(tmp_path: Path) -> None:
    """The report states the frame, the baseline it is all relative to, and one row per swap."""
    gate = SeatSwap(_frame(tmp_path), PARENTS)
    lines = gate.lines(gate.table([("child", "p0")]))
    assert any("champion proxy" in line for line in lines)
    assert any("parents mean pairwise corr" in line for line in lines)
    assert any(line.startswith("child") for line in lines)


def test_compositions(tmp_path: Path) -> None:
    """Every subset applied at once, against the sum of its singles — one row per k."""
    gate = SeatSwap(_frame(tmp_path), PARENTS)
    lines = gate.compositions([("child", "p0"), ("copycat", "p3")])
    assert len(lines) == 3, "a header and one row per k"
    assert lines[1].split()[1] == "2", "two single swaps at k=1"
    assert lines[2].split()[1] == "1", "one way to apply both"
