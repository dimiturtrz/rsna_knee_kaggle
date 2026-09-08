"""One candidate priced into a named pool off holdout records: the row relation is the whole difficulty."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from core.findings import TARGETS
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.evaluation.seat_add import BOARD_RESOLVABLE, Relation, SeatAdd, SeatAddRow
from kneemri.runs.holdout_record import HoldoutRecord

POOL_STUDIES = ["s1", "s2", "s3", "s4", "s5"]
GRADES = [0.94, 0.68, 0.68, 0.08, 0.94]
"""Rungs on both sides of both cuts, so every column has two classes at 0.75 and at the midpoint alike."""

SCORES = [0.9, 0.5, 0.4, 0.1, 0.8]


def _predictions(values: list[float]) -> NDArray[np.float64]:
    """The same prediction column repeated across all twelve findings."""
    return np.tile(np.array(values, dtype=np.float64)[:, None], (1, len(TARGETS)))


def _truth(values: list[float]) -> NDArray[np.float32]:
    """The same graded truth column repeated across all twelve findings."""
    return np.tile(np.array(values, dtype=np.float32)[:, None], (1, len(TARGETS)))


def _member(directory: Path, name: str, studies: list[str], grades: list[float] | None = None) -> Path:
    """A member whose record covers exactly the studies named, ranked in the order they are given."""
    rows = [POOL_STUDIES.index(study) if study in POOL_STUDIES else 0 for study in studies]
    graded = grades if grades is not None else [GRADES[row] for row in rows]
    path = directory / f"{name}.pt"
    HoldoutRecord(path).write(studies, _predictions([SCORES[row] for row in rows]), _truth(graded), epoch=7)
    return path


def _pool(directory: Path) -> list[Path]:
    """Two seated members, both on the full frame."""
    return [_member(directory, "seat1", POOL_STUDIES), _member(directory, "seat2", POOL_STUDIES)]


def test_add() -> None:
    """The add is the difference of the two macros, and nothing is subtracted from anything else."""
    row = SeatAddRow("c", Relation.NESTED, 12, 896, base=0.9032, with_candidate=0.9033, solo=0.81, ruler="ledger75")
    assert row.add() == pytest.approx(0.0001)


def test_clears() -> None:
    """The bar is measured, and the add that priced `ourtrunk44` sits under it by two orders."""
    fields = {"pool_size": 12, "rows": 896, "base": 0.0, "solo": 0.8, "ruler": "ledger75"}
    assert SeatAddRow("c", Relation.EXACT, with_candidate=BOARD_RESOLVABLE * 2, **fields).clears()
    assert not SeatAddRow("c", Relation.EXACT, with_candidate=BOARD_RESOLVABLE / 2, **fields).clears()


def test_reading(tmp_path: Path) -> None:
    """The kept arrays come back under their own study names."""
    reading = SeatAdd().reading(_member(tmp_path, "m1", POOL_STUDIES))
    assert reading.studies == POOL_STUDIES


def test_reading_refuses_a_member_with_no_record(tmp_path: Path) -> None:
    """A dark arm held nothing out, and no add over it can ever exist."""
    with pytest.raises(FileNotFoundError, match="no holdout record"):
        SeatAdd().reading(tmp_path / "dark.pt")


def test_shared(tmp_path: Path) -> None:
    """The shared frame is what EVERY reading covers, in an order two runs of this agree on."""
    instrument = SeatAdd()
    readings = [instrument.reading(path) for path in (_member(tmp_path, "wide", POOL_STUDIES),)]
    readings.append(instrument.reading(_member(tmp_path, "narrow", ["s3", "s1"])))
    assert instrument.shared(readings) == ["s1", "s3"]


def test_relation(tmp_path: Path) -> None:
    """Covering the frame exactly, sitting inside it, and spilling outside it are three findings."""
    instrument = SeatAdd()
    exact = instrument.reading(_member(tmp_path, "exact", POOL_STUDIES))
    nested = instrument.reading(_member(tmp_path, "nested", POOL_STUDIES[:4]))
    partial = instrument.reading(_member(tmp_path, "partial", ["s1", "s2", "outside"]))
    assert instrument.relation(exact, POOL_STUDIES) is Relation.EXACT
    assert instrument.relation(nested, POOL_STUDIES) is Relation.NESTED
    assert instrument.relation(partial, POOL_STUDIES) is Relation.PARTIAL


def test_stacked(tmp_path: Path) -> None:
    """Predictions are joined by study NAME, so a reading written in another order still aligns."""
    instrument = SeatAdd()
    forward = instrument.reading(_member(tmp_path, "forward", POOL_STUDIES))
    backward = instrument.reading(_member(tmp_path, "backward", list(reversed(POOL_STUDIES))))
    stacked = instrument.stacked([forward, backward], ["s2", "s4"])
    assert np.allclose(stacked[0], stacked[1])
    assert stacked[0][:, 0] == pytest.approx([SCORES[1], SCORES[3]])


def test_truth(tmp_path: Path) -> None:
    """The one truth table both readings agree on, over exactly the studies asked for."""
    instrument = SeatAdd()
    readings = [instrument.reading(path) for path in _pool(tmp_path)]
    assert instrument.truth(readings, ["s1", "s4"])[:, 0] == pytest.approx([GRADES[0], GRADES[3]])


def test_truth_refuses_two_rulers(tmp_path: Path) -> None:
    """Two records disagreeing on a shared study were scored against different truths, not one."""
    instrument = SeatAdd()
    theirs = _member(tmp_path, "theirs", POOL_STUDIES, grades=[0.08, 0.94, 0.68, 0.68, 0.94])
    readings = [instrument.reading(_member(tmp_path, "ours", POOL_STUDIES)), instrument.reading(theirs)]
    with pytest.raises(ValueError, match="different truth table"):
        instrument.truth(readings, POOL_STUDIES)


def test_macro(tmp_path: Path) -> None:
    """The pool's macro is this instrument's own vote and cut over the stacked arrays."""
    instrument = SeatAdd()
    readings = [instrument.reading(path) for path in _pool(tmp_path)]
    combined = instrument.vote.combine(instrument.stacked(readings, POOL_STUDIES))
    expected = instrument.scores.mean(combined, instrument.truth(readings, POOL_STUDIES))
    assert instrument.macro(readings, POOL_STUDIES) == pytest.approx(expected)


def test_price(tmp_path: Path) -> None:
    """A nested candidate narrows the question, and the base is recomputed on the narrowed rows."""
    row = SeatAdd().price(_pool(tmp_path), _member(tmp_path, "cand", POOL_STUDIES[:4]))
    assert row.relation is Relation.NESTED
    assert (row.rows, row.pool_size, row.candidate, row.ruler) == (4, 2, "cand", "ledger75")
    assert row.add() == pytest.approx(row.with_candidate - row.base)


def test_price_refuses_a_partial_overlap(tmp_path: Path) -> None:
    """A candidate holding studies the pool does not leaves a third question, and no seat."""
    with pytest.raises(ValueError, match="third question"):
        SeatAdd().price(_pool(tmp_path), _member(tmp_path, "spill", ["s1", "s2", "outside"]))


def test_price_names_the_cut_it_was_read_under(tmp_path: Path) -> None:
    """A macro is not comparable across cuts, so the row carries the ruler that produced it."""
    row = SeatAdd(LabelScores(MIDPOINT)).price(_pool(tmp_path), _member(tmp_path, "mid", POOL_STUDIES))
    assert row.ruler == "cut0.5"
