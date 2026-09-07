"""The swap proxy over a pool built in memory: the table is read along the COUNT of distilled seats.

Every array here is synthetic and seeded. The module's own reading is a statement about real distilled
twins; what these tests hold is the arithmetic that turns a pool into that statement — which seat a
configuration seats, what disagreement means when two members are identical, and that the all-originals
row is the baseline every other row is a delta against.
"""

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.scores import LabelScores
from kneemri.evaluation.swap_proxy import Configuration, Seat, SwapProxy

SEED = 2026
STUDIES = 24


def _pool() -> Pool:
    """Four members over one truth: two originals, and a distilled twin of each.

    The twins are the originals plus a small fixed perturbation, so a swap moves the score a little and
    the disagreement a lot — which is the pair of quantities every row of the table reports.
    """
    generator = np.random.default_rng(SEED)
    truth = (generator.random((STUDIES, len(TARGETS))) > 0.5).astype(np.float32)
    signal = truth + generator.normal(0.0, 0.4, truth.shape)
    predictions = [
        signal + generator.normal(0.0, 0.2, truth.shape),
        signal + generator.normal(0.0, 0.2, truth.shape),
    ]
    predictions += [one + 0.05 for one in predictions]
    return Pool(names=["a", "b", "a_df", "b_df"], predictions=predictions, truth=truth)


def _proxy() -> SwapProxy:
    """The two seats those four members make."""
    return SwapProxy([Seat("a", "a_df"), Seat("b", "b_df")], _pool())


def test_parsed() -> None:
    """`original=distilled`, refused rather than guessed at when it is not that shape."""
    assert Seat.parsed("band_dense=oof_teacher") == Seat("band_dense", "oof_teacher")
    for malformed in ("band_dense", "=oof_teacher", "band_dense="):
        with pytest.raises(ValueError, match="a pair is"):
            Seat.parsed(malformed)


def test_named() -> None:
    """Which member a seat contributes under one configuration."""
    seat = Seat("band_dense", "oof_teacher")
    assert seat.named(swapped=False) == "band_dense"
    assert seat.named(swapped=True) == "oof_teacher"


def test_count() -> None:
    """How many seats are distilled — the axis the table is read along."""
    assert Configuration(swapped=(False, True, True), score=0.5, per_label={}, disagreement=0.0).count() == 2
    assert Configuration(swapped=(False, False), score=0.5, per_label={}, disagreement=0.0).count() == 0


def test_line() -> None:
    """A row states its count, its score, its delta against the originals and the members it seats."""
    seats = [Seat("a", "a_df"), Seat("b", "b_df")]
    line = Configuration(swapped=(False, True), score=0.86, per_label={}, disagreement=0.02).line(seats, 0.88)
    assert "1/2 distilled" in line
    assert "-0.0200" in line, "a row that cannot show a loss is not a row a decision can be read off"
    assert "a, b_df" in line


def test_members() -> None:
    """The pool positions one configuration seats, in the caller's seat order."""
    proxy = _proxy()
    assert proxy.members((False, False)) == [0, 1]
    assert proxy.members((True, True)) == [2, 3]
    assert proxy.members((True, False)) == [2, 1]


def test_disagreement() -> None:
    """Mean pairwise absolute difference — zero for one member, and zero for a member against itself."""
    proxy = _proxy()
    assert proxy.disagreement([0]) == 0.0
    assert proxy.disagreement([1, 1]) == 0.0
    # THE TWIN IS ITS ORIGINAL PLUS 0.05 EVERYWHERE, so the mean absolute difference is exactly that.
    assert proxy.disagreement([0, 2]) == pytest.approx(0.05)


def test_scored() -> None:
    """One configuration's mean AUC, its per-label table and its internal disagreement."""
    configuration = _proxy().scored(LabelScores(), (False, True))
    assert configuration.swapped == (False, True)
    assert set(configuration.per_label) == set(TARGETS)
    assert configuration.score == pytest.approx(float(np.nanmean(list(configuration.per_label.values()))))
    assert configuration.disagreement > 0.0


def test_every() -> None:
    """All 2^n configurations, ORIGINALS FIRST — every later row is a delta against that one."""
    configurations = _proxy().every(LabelScores())
    assert len(configurations) == 4
    assert configurations[0].swapped == (False, False)
    assert configurations[-1].swapped == (True, True)
    assert [one.count() for one in configurations] == [0, 1, 1, 2]


def test_lines() -> None:
    """The report names both endpoints, ranks the halves, and prints the per-label swap table."""
    lines = _proxy().lines(LabelScores())
    assert any("ALL ORIGINALS" in line for line in lines)
    assert any("FULL SWAP" in line for line in lines)
    assert any("BEST configuration" in line for line in lines)
    halves = [line for line in lines if "1/2 distilled" in line]
    assert len(halves) == 2, "both intermediate configurations have to be visible, not just the better one"
    assert [float(line.split()[2]) for line in halves] == sorted(
        [float(line.split()[2]) for line in halves], reverse=True
    ), "the halves are ranked best first, which is what makes the collective-vs-single reading possible"
    assert any(line.startswith("  MCL") and "<- watched" in line for line in lines)
