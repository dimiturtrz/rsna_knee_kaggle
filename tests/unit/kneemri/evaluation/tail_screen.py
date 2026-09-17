"""The shelf ranked by tail decorrelation — the axis solo score cannot see."""

from collections.abc import Sequence
from pathlib import Path
from typing import override

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from kneemri.evaluation.study_attribution import AttributionCorpus
from kneemri.evaluation.study_errors import StudyDamage
from kneemri.evaluation.tail_screen import ShelfScreen, TailOverlap, TailScreen


def _damage(study: str, damage: float) -> StudyDamage:
    """One ranking row, carrying only the two fields the tail cut reads."""
    return StudyDamage(
        study=study, damage=damage, positives=3, worst_finding="ACL", worst_damage=damage, over=damage, under=0.0
    )


def _ranking(order: list[str]) -> list[StudyDamage]:
    """A ranking worst-first, which is the order `StudyErrors.ranked` hands back."""
    return [_damage(study, float(len(order) - index)) for index, study in enumerate(order)]


class _Shelf(AttributionCorpus):
    """A corpus whose members are handed in rather than read, so no test touches a data root."""

    @override
    def ranked(
        self, run: str
    ) -> tuple[
        list[StudyDamage],
        Float[NDArray[np.float64], "studies findings"],
        list[str],
        Float[NDArray[np.floating], "studies findings"],
    ]:
        studies = [f"s{index}" for index in range(10)]
        return _ranking(studies), np.zeros((10, 12)), studies, np.zeros((10, 12))

    @override
    def members(
        self, run: str, studies: list[str], wanted: int, named: Sequence[str] = ()
    ) -> dict[str, list[StudyDamage]]:
        return {
            "twin": _ranking([f"s{index}" for index in range(10)]),
            "stranger": _ranking([f"s{index}" for index in reversed(range(10))]),
        }


def _screen() -> TailScreen:
    """The screen over the stub shelf."""
    return TailScreen(_Shelf(models=Path("models"), raw=Path("raw"), processed=Path("processed")))


def test_line() -> None:
    """A row carries the COUNT the fraction came from, so a small tail cannot hide behind a ratio."""
    assert TailOverlap(member="m", overlap=0.25, shared=1, tail=4).line().startswith("0.250    1/  4  m")


def test_lines() -> None:
    """A member well under the shelf mean is a seat the solo screen could not see."""
    rows = (
        TailOverlap(member="odd", overlap=0.15, shared=15, tail=90),
        TailOverlap(member="usual", overlap=0.70, shared=63, tail=90),
    )
    assert "CANDIDATE — odd" in "\n".join(ShelfScreen(run="ref", rows=rows, skipped=0).lines())


def test_lines_closes_the_lane_when_the_floor_sits_at_the_mean() -> None:
    """A shelf whose minimum is its mean fails the same studies throughout — closure on mechanism."""
    rows = (
        TailOverlap(member="a", overlap=0.62, shared=56, tail=90),
        TailOverlap(member="b", overlap=0.64, shared=58, tail=90),
    )
    assert "CLOSED" in "\n".join(ShelfScreen(run="ref", rows=rows, skipped=0).lines())


def test_lines_says_so_when_nothing_aligns() -> None:
    """An empty shelf is a denominator statement, not a ranking — it may not read as closure."""
    assert "no member" in ShelfScreen(run="ref", rows=(), skipped=7).lines()[0]


def test_shelved(tmp_path: Path) -> None:
    """The universe is every checkpoint with a holdout record, minus the reference itself."""
    for name in ("ref", "other"):
        (tmp_path / f"{name}.holdout.npz").write_bytes(b"")
    screen = TailScreen(_Shelf(models=tmp_path, raw=tmp_path, processed=tmp_path))
    assert screen.shelved("ref") == {"other"}


def test_screen() -> None:
    """A member sharing the tail reads 1.0 and one whose tail is the other end reads 0.0, ranked first."""
    screen = _screen().screen("ref")
    assert [row.member for row in screen.rows] == ["stranger", "twin"]
    assert screen.rows[0].overlap == 0.0
    assert screen.rows[1].overlap == 1.0
