"""The second ledger ruler: ten columns recut at 0.75, two held at the midpoint, both numbers kept."""

from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from core.findings import TARGETS
from kneemri.evaluation.ledger75 import MIDPOINT_COLUMNS, Ledger75Row, LedgerSeventyFive, Ruler
from kneemri.evaluation.member_pool import MemberDeclines
from kneemri.evaluation.scores import LabelScores
from kneemri.runs.holdout_record import HoldoutRecord

STUDIES = ["s1", "s2", "s3", "s4"]
GRADES = [0.94, 0.68, 0.68, 0.08]
"""Two of the reader's rungs on each side of the recut: 0.68 is positive at the midpoint and not at 0.75."""

RECUT_COLUMN = TARGETS[TARGETS.index("Effusion")]
HELD_COLUMN = MIDPOINT_COLUMNS[0]


def _truth() -> NDArray[np.float32]:
    """The same graded column repeated across all twelve findings."""
    return np.tile(np.array(GRADES, dtype=np.float32)[:, None], (1, len(TARGETS)))


def _member(directory: Path, name: str, ranked: str) -> Path:
    """A member that ranks `ranked` by the 0.94 row alone and every other finding backwards."""
    truth = _truth()
    predictions = np.tile(np.array([0.1, 0.9, 0.9, 0.9])[:, None], (1, len(TARGETS)))
    predictions[:, TARGETS.index(ranked)] = np.array([0.9, 0.2, 0.2, 0.1])
    path = directory / f"{name}.pt"
    HoldoutRecord(path).write(STUDIES, predictions, truth, epoch=5)
    return path


def test_reading(tmp_path: Path) -> None:
    """The kept arrays come back under their study names and the epoch they were written with."""
    reading = LedgerSeventyFive().reading(_member(tmp_path, "m1", RECUT_COLUMN))
    assert reading.studies == STUDIES
    assert reading.epoch == 5


def test_reading_refuses_a_member_with_no_record(tmp_path: Path) -> None:
    """A dark arm held nothing out, and the ruler must say so rather than price nothing."""
    with pytest.raises(FileNotFoundError, match="no holdout record"):
        LedgerSeventyFive().reading(tmp_path / "dark.pt")


def test_ledger_seventy_five_scorer(tmp_path: Path) -> None:
    """The mixed cut as a plain scorer: a caller holding arrays reads exactly what `macro` reads."""
    ruler = LedgerSeventyFive()
    reading = ruler.reading(_member(tmp_path, "m6", RECUT_COLUMN))
    scorer = ruler.scorer()
    assert scorer.cut_for(RECUT_COLUMN) == pytest.approx(0.75)
    assert scorer.cut_for(HELD_COLUMN) == pytest.approx(0.5)
    assert scorer.mean(reading.predictions, reading.truth) == pytest.approx(ruler.macro(reading))


def test_per_label(tmp_path: Path) -> None:
    """A member that ranks only the 0.94 row scores 1.0 where the cut is 0.75 and worse where it is not."""
    reading = LedgerSeventyFive().reading(_member(tmp_path, "m2", RECUT_COLUMN))
    per_label = LedgerSeventyFive().per_label(reading)
    assert per_label[RECUT_COLUMN] == pytest.approx(1.0)
    assert per_label[HELD_COLUMN] < 1.0


def test_macro(tmp_path: Path) -> None:
    """The macro is the NaN-skipping mean of exactly the columns `per_label` returns."""
    ruler = LedgerSeventyFive()
    reading = ruler.reading(_member(tmp_path, "m3", RECUT_COLUMN))
    assert ruler.macro(reading) == pytest.approx(float(np.nanmean(list(ruler.per_label(reading).values()))))


def test_row(tmp_path: Path) -> None:
    """A row is the pair: the midpoint macro the ledger holds and the recut macro beside it."""
    ruler = LedgerSeventyFive()
    path = _member(tmp_path, "m4", RECUT_COLUMN)
    row = ruler.row(path)
    reading = ruler.reading(path)
    midpoint = LabelScores().per_label(reading.predictions, reading.truth)
    assert row.name == "m4"
    assert row.studies == len(STUDIES)
    assert row.ledger == pytest.approx(float(np.nanmean(list(midpoint.values()))))
    assert row.ledger75 == pytest.approx(ruler.macro(reading))


def test_moved(tmp_path: Path) -> None:
    """The move is one member's own re-pricing, not a comparison against another member."""
    row = LedgerSeventyFive().row(_member(tmp_path, "m5", RECUT_COLUMN))
    assert row.moved() == pytest.approx(row.ledger75 - row.ledger)


def _binary_member(directory: Path, name: str) -> Path:
    """A member whose truth is 0/1 only, so the two cuts read the identical array."""
    truth = np.tile(np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)[:, None], (1, len(TARGETS)))
    predictions = np.tile(np.array([0.9, 0.2, 0.8, 0.1])[:, None], (1, len(TARGETS)))
    path = directory / f"{name}.pt"
    HoldoutRecord(path).write(STUDIES, predictions, truth, epoch=5)
    return path


def test_between(tmp_path: Path) -> None:
    """Only the truth cells the two rulers disagree about are counted, over the whole array."""
    ruler = LedgerSeventyFive()
    reading = ruler.reading(_member(tmp_path, "m6", RECUT_COLUMN))

    assert ruler.between(reading) == 2 * len(TARGETS), "the two 0.68 rows, on every finding"


def test_recut_blind(tmp_path: Path) -> None:
    """A record with no truth between the cuts is priced identically by both, so its pair says nothing."""
    ruler = LedgerSeventyFive()

    graded = ruler.row(_member(tmp_path, "m7", RECUT_COLUMN))
    binary = ruler.row(_binary_member(tmp_path, "probe"))

    assert not graded.recut_blind(), "0.68 rows are exactly what the recut moves"
    assert binary.recut_blind(), "0/1 truth cannot land between the cuts"
    assert binary.moved() == pytest.approx(0.0), "which is why its zero is construction, not measurement"


def test_table(tmp_path: Path) -> None:
    """A dark arm beside the records must not take the sweep down, and the order is LEDGER75's."""
    strong = _member(tmp_path, "strong", RECUT_COLUMN)
    weak = _member(tmp_path, "weak", HELD_COLUMN)
    rows = LedgerSeventyFive().table([strong, weak, tmp_path / "dark.pt"])
    assert [row.name for row in rows] == ["strong", "weak"]


def _short_member(directory: Path, name: str) -> Path:
    """A member read on a DIFFERENT, smaller study set — a second population, scoring perfectly."""
    truth = np.tile(np.array([1.0, 0.0], dtype=np.float32)[:, None], (1, len(TARGETS)))
    predictions = np.tile(np.array([0.99, 0.01])[:, None], (1, len(TARGETS)))
    path = directory / f"{name}.pt"
    HoldoutRecord(path).write(["s9", "s8"], predictions, truth, epoch=5)
    return path


def test_population(tmp_path: Path) -> None:
    """The digest is the study SET, so record order cannot split one population into two."""
    ruler = LedgerSeventyFive()
    one = ruler.reading(_member(tmp_path, "p1", RECUT_COLUMN))
    two = ruler.reading(_member(tmp_path, "p2", RECUT_COLUMN))
    assert ruler.population(one) == ruler.population(two)
    assert ruler.population(one) != ruler.population(ruler.reading(_short_member(tmp_path, "p3")))


def test_population_separates_a_moved_label_table(tmp_path: Path) -> None:
    """Same studies graded by another table is another question, so it is another population."""
    ruler = LedgerSeventyFive()
    ours = ruler.reading(_member(tmp_path, "corpus_table", RECUT_COLUMN))
    moved = _member(tmp_path, "moved_table", RECUT_COLUMN)
    kept = ruler.reading(moved)
    HoldoutRecord(moved).write(STUDIES, kept.predictions, _truth() * 0.5, epoch=5)
    assert ruler.population(ours) != ruler.population(ruler.reading(moved))


def test_population_ignores_record_order(tmp_path: Path) -> None:
    """Shuffling the rows of one record leaves the truth digested in the same study order."""
    ruler = LedgerSeventyFive()
    straight = ruler.reading(_member(tmp_path, "straight", RECUT_COLUMN))
    shuffled = _member(tmp_path, "shuffled", RECUT_COLUMN)
    order = [3, 1, 0, 2]
    HoldoutRecord(shuffled).write(
        [STUDIES[index] for index in order],
        straight.predictions[order],
        _truth()[order],
        epoch=5,
    )
    assert ruler.population(straight) == ruler.population(ruler.reading(shuffled))


def test_table_keeps_populations_whole(tmp_path: Path) -> None:
    """A higher score on another study set may not outrank the majority population — it sorts after it."""
    rows = LedgerSeventyFive().table(
        [
            _short_member(tmp_path, "perfect_elsewhere"),
            _member(tmp_path, "ours_a", RECUT_COLUMN),
            _member(tmp_path, "ours_b", HELD_COLUMN),
        ]
    )
    assert rows[-1].name == "perfect_elsewhere", "it scores 1.0 and still may not lead"
    assert rows[0].studies == len(STUDIES)


def test_void(tmp_path: Path) -> None:
    """A member the declines file calls void is MARKED, never dropped and never silently ranked."""
    declines = MemberDeclines("insample.pt  # LEDGER75 0.9425 - void, in-sample.\nhonest.pt  # below the floor.\n")
    ruler = LedgerSeventyFive(declines=declines)
    assert ruler.row(_member(tmp_path, "insample", RECUT_COLUMN)).void
    assert not ruler.row(_member(tmp_path, "honest", RECUT_COLUMN)).void
    assert not LedgerSeventyFive().row(_member(tmp_path, "unjudged", RECUT_COLUMN)).void


def test_price(tmp_path: Path) -> None:
    """A checkpoint carrying no holdout gets this row's own midpoint macro; one carrying a number keeps it."""
    path = _member(tmp_path, "m9", RECUT_COLUMN)
    row = LedgerSeventyFive().row(path)

    torch.save({"model": {}, "holdout": float("nan")}, path)
    assert row.price(path), "a NaN price is what a weight soup arrives with"
    assert float(torch.load(path, weights_only=True)["holdout"]) == pytest.approx(row.ledger)

    torch.save({"model": {}, "holdout": 0.5}, path)
    assert not row.price(path), "a recorded price is never silently replaced by a ruler"
    assert float(torch.load(path, weights_only=True)["holdout"]) == pytest.approx(0.5)


def _plain(name: str, population: str = "p1", *, void: bool = False) -> Ledger75Row:
    return Ledger75Row(
        name=name, ledger=0.8, ledger75=0.85, epoch=3, studies=12, between_cut=4, population=population, void=void
    )


def test_line() -> None:
    """A row prints its pair and move, and carries a marker only when it is VOID or BLIND."""
    assert _plain("m").line().split() == ["m", "0.8000", "0.8500", "+0.0500", "3", "12"]
    assert "VOID" in _plain("m", void=True).line()
    assert "BLIND" in Ledger75Row(name="m", ledger=0.8, ledger75=0.8, epoch=3, studies=12, between_cut=0).line()


def test_lines() -> None:
    """One header, one divider per population, one line per row, and the count."""
    lines = LedgerSeventyFive.lines([_plain("a"), _plain("b"), _plain("c", "p2")])
    assert sum(line.startswith("-- population") for line in lines) == 2
    assert len(lines) == 1 + 2 + 3 + 1
    assert lines[-1] == "3 member(s) priced under both rulers"


def test_priced(tmp_path: Path) -> None:
    """A VOID row is never written, and a row with no checkpoint on disk is skipped rather than raised on."""
    torch.save({"model": {}, "holdout": float("nan")}, tmp_path / "m.pt")
    rows = [_plain("m"), _plain("gone"), _plain("insample", void=True)]
    assert LedgerSeventyFive.priced(rows, tmp_path) == "priced 1 checkpoint(s) that carried no holdout: m"
    assert LedgerSeventyFive.priced([], tmp_path).endswith("none")


def test_refusal() -> None:
    """VOID always refuses; a span of populations refuses only when a comparison was asked for."""
    spanning = [_plain("a"), _plain("b", "p2")]
    assert not LedgerSeventyFive.refusal(spanning, Path("declines"), compared=False)
    assert LedgerSeventyFive.refusal(spanning, Path("declines"), compared=True).startswith("--only spans 2")
    assert "VOID" in LedgerSeventyFive.refusal([_plain("x", void=True)], Path("declines"), compared=False)
    assert not LedgerSeventyFive.refusal([_plain("a")], Path("declines"), compared=True)


def test_column():
    """Each ruler records into its own field, and the midpoint keeps the field every older row carries."""
    assert Ruler.MIDPOINT.column == "offline"
    assert Ruler.LEDGER75.column == "offline75"
    assert len({ruler.column for ruler in Ruler}) == len(Ruler)


def test_ruler_scorer():
    """The midpoint reads every finding at 0.5; the recut is the instrument's own mixed cut, not a copy."""
    midpoint, recut, instrument = Ruler.MIDPOINT.scorer(), Ruler.LEDGER75.scorer(), LedgerSeventyFive().scorer()
    assert [midpoint.cut_for(target) for target in TARGETS] == [LabelScores().cut_for(target) for target in TARGETS]
    assert [recut.cut_for(target) for target in TARGETS] == [instrument.cut_for(target) for target in TARGETS]
    assert recut.cut_for(MIDPOINT_COLUMNS[0]) == midpoint.cut_for(MIDPOINT_COLUMNS[0])
    assert recut.cut != midpoint.cut
    assert Ruler.LEDGER75.scorer() is not recut, "a fresh scorer per call, never a shared instance"
