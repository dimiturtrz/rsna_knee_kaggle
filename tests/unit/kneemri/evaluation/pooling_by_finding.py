"""Unit cover for the pooling-by-finding contrast.

The equivalence classes are GROUPING and ARITHMETIC over records this test writes itself: a run with a
stated pooling rule against one without, a family ranked by its own macro, a per-column mean, a gap that
is right minus left, and the three routing totals. Whether the focal/diffuse split actually sorts the
real pool is a measurement over recorded holdouts and belongs to the module's run, never here.
"""

from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.pooling_by_finding import DIFFUSE, FOCAL, POOLING, PoolingByFinding, PoolingContrast
from kneemri.runs.holdout_record import HoldoutRecord
from kneemri.runs.run_metrics import RunMetrics

STUDIES = 80
STRONG = 3.0
"""Signal added to a member meant to outrank the others — separable at this size, no seed dependence."""


def record(holdouts: Path, stem: str, rule: str | None, lift: float = 0.0) -> None:
    """One member on disk: a holdout record, and a metrics file that may or may not state a pooling rule."""
    generator = np.random.default_rng(abs(hash(stem)) % 1000)
    truth = generator.integers(0, 2, size=(STUDIES, len(TARGETS))).astype(np.float32)
    predictions = (generator.normal(size=(STUDIES, len(TARGETS))) + truth * (1.0 + lift)).astype(np.float64)
    studies = [f"study{index:04d}" for index in range(STUDIES)]
    HoldoutRecord(holdouts / f"{stem}.pt").write(studies, predictions, truth, epoch=3)
    metrics = RunMetrics(holdouts / f"{stem}.pt")
    metrics.describe({POOLING: rule} if rule is not None else {"epochs": "12"})


@pytest.fixture
def pool(tmp_path: Path) -> Path:
    """Two members under each of two rules, plus one run that never stated a rule at all."""
    record(tmp_path, "max_a", "max", lift=STRONG)
    record(tmp_path, "max_b", "max")
    record(tmp_path, "labelwise_a", "labelwise", lift=STRONG)
    record(tmp_path, "labelwise_b", "labelwise")
    record(tmp_path, "unstamped", None)
    return tmp_path


def test_rule(pool: Path) -> None:
    """A stamped run reports its rule; one whose recipe never named the key reports None."""
    assert PoolingByFinding(pool).rule("max_a") == "max"
    assert PoolingByFinding(pool).rule("unstamped") is None


def test_families(pool: Path) -> None:
    """Runs group by their stated rule, and the unstamped one joins no family."""
    grouped = PoolingByFinding(pool).families()
    assert grouped == {"max": ["max_a", "max_b"], "labelwise": ["labelwise_a", "labelwise_b"]}


def test_read(pool: Path) -> None:
    """A member's record comes back with the arrays it was written from."""
    reading = PoolingByFinding(pool).read("max_a")
    assert reading.predictions.shape == (STUDIES, len(TARGETS))
    assert reading.epoch == 3


def test_columns(pool: Path) -> None:
    """Twelve findings, one LEDGER75 value each, all of them real numbers on [0, 1]."""
    columns = PoolingByFinding(pool).columns("max_a")
    assert set(columns) == set(TARGETS)
    assert all(0.0 <= value <= 1.0 for value in columns.values())


def test_strongest(pool: Path) -> None:
    """A family is cut to its best by macro — the lifted member first, and `top` is a ceiling."""
    reader = PoolingByFinding(pool)
    assert reader.strongest(["max_b", "max_a"], top=2) == ["max_a", "max_b"]
    assert reader.strongest(["max_b", "max_a"], top=1) == ["max_a"]


def test_family_columns(pool: Path) -> None:
    """A family's column is the mean of its members' columns on that finding."""
    reader = PoolingByFinding(pool)
    family = reader.family_columns(["max_a", "max_b"])
    for target in TARGETS:
        members = [reader.columns("max_a")[target], reader.columns("max_b")[target]]
        assert family[target] == pytest.approx(float(np.mean(members)))


def test_contrast(pool: Path) -> None:
    """One row per finding, each carrying the split it was assigned before either family was read."""
    rows = PoolingByFinding(pool).contrast("max", "labelwise", top=2)
    assert [row.finding for row in rows] == list(TARGETS)
    assert {row.finding for row in rows if row.focal} == set(FOCAL)
    assert {row.finding for row in rows if not row.focal} == set(DIFFUSE)


def test_contrast_refuses_an_absent_family(pool: Path) -> None:
    """A rule no run on disk was fitted under is a missing family, not an empty one."""
    with pytest.raises(KeyError, match="attend"):
        PoolingByFinding(pool).contrast("max", "attend")


def test_gap() -> None:
    """The gap is right minus left, so a positive gap favours the right-hand family."""
    assert PoolingContrast("ACL", focal=True, left=0.80, right=0.85).gap() == pytest.approx(0.05)


def test_sorts() -> None:
    """The two means are taken within their own split and nowhere across it."""
    rows = [PoolingContrast(target, target in FOCAL, 0.80, 0.90 if target in DIFFUSE else 0.78) for target in TARGETS]
    focal, diffuse = PoolingByFinding(Path()).sorts(rows)
    assert focal == pytest.approx(-0.02)
    assert diffuse == pytest.approx(+0.10)


def test_routed() -> None:
    """Blind routing takes left on focal and right on diffuse; the oracle takes the winner everywhere."""
    rows = [PoolingContrast(target, target in FOCAL, 0.80, 0.90 if target in DIFFUSE else 0.78) for target in TARGETS]
    worth = PoolingByFinding(Path()).routed(rows)
    assert worth["left"] == pytest.approx(0.80)
    assert worth["right"] == pytest.approx(0.84)
    assert worth["blind"] == pytest.approx(0.85)
    assert worth["oracle"] == pytest.approx(0.85)
