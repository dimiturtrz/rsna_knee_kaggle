"""`RankStability` — the ordering of several arms under resampling of the studies.

Every fixture here is built in the test. A synthetic pool is what makes the assertions decidable: with
one arm constructed to dominate every study, the first-rank share MUST be 1.0, and with two arms carrying
identical predictions the tau between their rulers MUST be 1.0. A real holdout could assert neither.
"""

from pathlib import Path

import numpy as np
import pytest

from kneemri.evaluation.pool import Pool
from kneemri.evaluation.rank_stability import (
    RESOLVED_SHARE,
    TARGET_COLUMNS,
    ArmStability,
    RankStability,
    SeriesAxes,
)
from kneemri.evaluation.scores import MIDPOINT

LABELS = 12
STUDIES = 200
RESAMPLES = 40


def _truth() -> np.ndarray:
    """Half positive on every column, so no column is ever undefined on the full set."""
    rows = np.zeros((STUDIES, LABELS), dtype=np.float32)
    rows[: STUDIES // 2, :] = 1.0
    return rows


def _pool() -> Pool:
    """Three arms: a perfect one, a noisy one, and a constant one that can never rank first."""
    truth = _truth()
    generator = np.random.default_rng(2026)
    perfect = truth.astype(np.float64) + generator.normal(0.0, 0.01, truth.shape)
    noisy = truth.astype(np.float64) + generator.normal(0.0, 0.60, truth.shape)
    flat = np.tile(np.linspace(0.0, 1.0, STUDIES)[:, None], (1, LABELS))
    return Pool(["perfect", "noisy", "flat"], [perfect, noisy, flat], truth)


def test_ruler() -> None:
    """The two truths the cut question compares are distinguishable by name."""
    stability = RankStability(_pool(), RESAMPLES)
    assert stability.ruler().cut == MIDPOINT
    assert stability.ruler(recut=True).cut > MIDPOINT
    assert stability.ruler(recut=True).name() == "ledger75"


def test_scored() -> None:
    """Every arm gets one macro, in pool order, and the constructed ordering is the one that comes back."""
    stability = RankStability(_pool(), RESAMPLES)
    values = stability.scored(stability.ruler())
    assert len(values) == 3
    assert values[0] > values[1] > values[2]


def test_defined() -> None:
    """A truth with both classes present on every column is a macro over all twelve."""
    stability = RankStability(_pool(), RESAMPLES)
    assert stability.defined(stability.ruler()) == [TARGET_COLUMNS] * 3


def test_order() -> None:
    """Rank 1 is the best score, and a NaN sinks rather than winning."""
    assert list(RankStability.order([0.9, 0.7, 0.8])) == [1, 3, 2]
    assert list(RankStability.order([float("nan"), 0.7])) == [2, 1]


def test_table() -> None:
    """An arm built to dominate every study holds rank 1 in every resample; a constant one in none."""
    stability = RankStability(_pool(), RESAMPLES)
    rows = {row.name: row for row in stability.table(stability.ruler())}
    assert rows["perfect"].rank == 1
    assert rows["perfect"].first_share == pytest.approx(1.0)
    assert rows["flat"].first_share == pytest.approx(0.0)
    assert rows["perfect"].mean_rank < rows["noisy"].mean_rank


def test_cut_agreement() -> None:
    """Two rulers over arms this far apart cannot reorder them, so the tau is total agreement."""
    assert RankStability(_pool(), RESAMPLES).cut_agreement() == pytest.approx(1.0)


def test_report() -> None:
    """The report names the denominator, the ruler, every arm, and the verdict."""
    lines = RankStability(_pool(), RESAMPLES).report(RankStability(_pool(), RESAMPLES).ruler())
    assert f"3 arm(s) over {STUDIES} shared studies" in lines[0]
    assert any("perfect" in line for line in lines)
    assert any("RESOLVED by this holdout" in line for line in lines)


def test_within() -> None:
    """A subgroup scores every arm over its own studies only, and the constructed ordering survives."""
    stability = RankStability(_pool(), RESAMPLES)
    keep = np.zeros(STUDIES, dtype=np.bool_)
    keep[: STUDIES // 4] = True
    keep[STUDIES // 2 : STUDIES // 2 + STUDIES // 4] = True
    values = stability.within(stability.ruler(), keep)
    assert len(values) == 3
    assert values[0] > values[2]


def test_subgroup_agreement() -> None:
    """Two halves of a pool this separable agree completely, and both the n and the tau are reported."""
    stability = RankStability(_pool(), RESAMPLES)
    odd = np.arange(STUDIES) % 2 == 1
    lines = stability.subgroup_agreement(stability.ruler(), {"even": ~odd, "odd": odd})
    assert any("n=  100" in line for line in lines)
    assert any(line.startswith("tau even vs odd: +1.0000") for line in lines)


def test_control() -> None:
    """The floor is two disjoint halves that together cover every study exactly once."""
    halves = RankStability(_pool(), RESAMPLES).control()
    first, second = halves["random half A"], halves["random half B"]
    assert int(first.sum()) == STUDIES // 2
    assert not (first & second).any()
    assert (first | second).all()


def test_read(tmp_path: Path) -> None:
    """Both axes come off one pass of the series table, keyed by study."""
    table = tmp_path / "train_series.csv"
    table.write_text(
        "StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane\n"
        "a,a1,1,1,Sagittal\na,a2,1,0,Axial\nb,b1,1,1,Coronal\n"
    )
    axes = SeriesAxes.read(table)
    assert axes.counts == {"a": 2, "b": 1}
    assert axes.suppressed == pytest.approx({"a": 0.5, "b": 1.0})


def test_groups() -> None:
    """Each axis splits at its own median, in the row order it was handed, and a missing study refuses."""
    axes = SeriesAxes({"a": 3, "b": 9}, {"a": 1.0, "b": 0.0})
    groups = axes.groups(["a", "b"])
    assert list(groups["series<=median"]) == [True, False]
    assert list(groups["suppression>median"]) == [True, False]
    with pytest.raises(ValueError, match="absent from the series table"):
        axes.groups(["a", "c"])


def test_resolved() -> None:
    """Only the top arm, and only above the legislated share, is an ordering rather than a coin flip."""
    assert ArmStability("a", 0.9, 1, RESOLVED_SHARE, 1.0, TARGET_COLUMNS).resolved()
    assert not ArmStability("a", 0.9, 1, RESOLVED_SHARE / 2, 1.0, TARGET_COLUMNS).resolved()
    assert not ArmStability("a", 0.9, 2, 1.0, 2.0, TARGET_COLUMNS).resolved()


def test_thin() -> None:
    """A macro over fewer than the board's twelve columns is named as a denominator difference."""
    assert ArmStability("a", 0.9, 1, 1.0, 1.0, TARGET_COLUMNS - 1).thin()
    assert not ArmStability("a", 0.9, 1, 1.0, 1.0, TARGET_COLUMNS).thin()
