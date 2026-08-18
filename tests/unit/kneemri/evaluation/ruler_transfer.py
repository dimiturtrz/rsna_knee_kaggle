"""Two arms scored against two label sources, on truth written by hand so each delta is known."""

import numpy as np

from kneemri.evaluation.ruler_transfer import RulerTransfer, Transfer
from kneemri.findings import TARGETS

STUDIES = 40


def _predictions(strength: float) -> np.ndarray:
    """A predictor that agrees with the truth below in proportion to `strength`."""
    ordered = np.linspace(0.0, 1.0, STUDIES, dtype=np.float64)
    noisy = np.where(np.arange(STUDIES) % 2 == 0, ordered, 1.0 - ordered)
    return np.tile((strength * ordered + (1.0 - strength) * noisy)[:, None], (1, len(TARGETS)))


def _truth() -> np.ndarray:
    """The back half positive on every finding, so the AUC of `_predictions` rises with strength."""
    truth = np.zeros((STUDIES, len(TARGETS)), dtype=np.float32)
    truth[STUDIES // 2 :, :] = 1.0
    return truth


def test_arms():
    """One arm per member count, and a count larger than the corpus is skipped rather than padded."""
    members = [np.full((4, len(TARGETS)), fill, dtype=np.float64) for fill in (0.1, 0.2)]

    arms = RulerTransfer().arms(members, sizes=(1, 2, 4))

    assert sorted(arms) == [1, 2], "four members were never mounted, so there is no four-member arm"
    assert np.allclose(arms[1], members[0]), "a one-member arm is that member, not a rank map of it"
    assert arms[2].shape == (4, len(TARGETS))


def test_gap():
    """The mean-over-findings difference, signed from weaker to stronger."""
    truth = _truth()
    transfer = RulerTransfer(resamples=8)

    rises = transfer.gap(_predictions(0.0), _predictions(1.0), truth)
    assert rises > 0.0

    assert transfer.gap(_predictions(1.0), _predictions(0.0), truth) == -rises, "the mirror pair, exactly"
    assert transfer.gap(_predictions(1.0), _predictions(1.0), truth) == 0.0


def test_interval():
    """A percentile band, and NaNs rather than a crash when no resample contributed."""
    low, high = RulerTransfer().interval([float(value) for value in range(101)])

    assert (round(low), round(high)) == (2, 98)
    assert all(np.isnan(bound) for bound in RulerTransfer().interval([]))


def test_transfer():
    """Both rulers see the same draw, and a ruler that agrees returns the same delta.

    The second ruler here is the first with two studies flipped: enough that it is a genuinely different
    label source, little enough that the arm ordering must survive it. A ratio near one is what "the
    ledger measures the right axis" looks like.
    """
    reported = _truth()
    annotated = reported.copy()
    annotated[[0, STUDIES - 1], :] = 1.0 - annotated[[0, STUDIES - 1], :]

    result = RulerTransfer(resamples=64).transfer(_predictions(0.0), _predictions(1.0), reported, annotated)

    assert result.reported > 0.0
    assert result.annotated > 0.0
    assert result.reported_interval[0] < result.reported < result.reported_interval[1]
    assert result.ratios > 0, "a delta this large never bootstraps below the floor"
    assert 0.5 < result.ratio < 1.5, "a ruler disagreeing on 2 of 40 studies returns most of the delta"


def test_weakened():
    """A blend toward a permutation of itself, monotone in how much is blended in."""
    truth = _truth()
    transfer = RulerTransfer(resamples=8)
    arm = _predictions(1.0) + np.linspace(0.0, 0.01, len(TARGETS))[None, :]

    intact = transfer.scores.mean(arm, truth)
    mild = transfer.scores.mean(transfer.weakened(arm, 0.1), truth)
    heavy = transfer.scores.mean(transfer.weakened(arm, 0.5), truth)

    assert intact > mild > heavy, "more permutation blended in is monotonically worse"
    assert np.allclose(transfer.weakened(arm, 0.0), arm), "and blending in none of it changes nothing"


def test_spread():
    """One row per degradation size, carrying the range ACROSS seeds rather than one seed's answer."""
    truth = _truth()
    arm = _predictions(1.0) + np.linspace(0.0, 0.01, len(TARGETS))[None, :]

    rows = RulerTransfer(resamples=8).spread(arm, truth, truth, shares=(0.2, 0.4), seeds=(11, 101, 2026))

    assert "across 3 permutation seeds" in rows[0]
    assert len(rows) == 3, "a header and one row per share"
    assert rows[2].strip().startswith("0.40")
    assert "+1.00" in rows[2], "the same ruler twice returns the whole delta, whatever the seed"


def test_ratio():
    """Undefined rather than enormous when the denominator is a delta of nearly nothing."""
    empty = (float("nan"), float("nan"))

    assert Transfer(0.02, 0.01, empty, empty, empty, 0).ratio == 0.5
    assert np.isnan(Transfer(0.0001, 0.01, empty, empty, empty, 0).ratio), "no ratio off a zero delta"


def test_lines():
    """Both deltas on one row, because the comparison between them is the finding."""
    band = (0.01, 0.03)
    pairs = {"1->2": Transfer(0.02, 0.01, band, band, (0.4, 0.6), 900)}

    rows = RulerTransfer(resamples=64).lines(pairs, studies=58)

    assert rows[0] == "58 studies scored twice; 64 paired resamples"
    assert "1->2" in rows[2]
    assert "+0.0200" in rows[2] and "+0.0100" in rows[2]
    assert "+0.50" in rows[2], "and the ratio the pair exists to produce"
