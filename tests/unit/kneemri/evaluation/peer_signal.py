"""The peer term is reported against the supervised one, and two identical members disagree about nothing."""

import numpy as np
import pytest

from kneemri.evaluation.peer_signal import PeerComparison, PeerSignal
from kneemri.runs.holdout_record import HoldoutReading

LABELS = 12
STUDIES = 40
IDENTICAL_TOLERANCE = 1e-9
"""Two members holding the same array must read a peer term of zero, not a small number."""


def _reading(names: list[str], seed: int) -> HoldoutReading:
    generator = np.random.default_rng(seed)
    return HoldoutReading(
        studies=names,
        predictions=generator.random((len(names), LABELS)),
        truth=generator.integers(0, 2, (len(names), LABELS)).astype(np.float32),
        epoch=seed,
        source="trained",
    )


def _names(count: int = STUDIES) -> list[str]:
    return [f"s{index}" for index in range(count)]


def test_ratio() -> None:
    """Divides the peer term by the supervised one, and reads zero rather than dividing by nothing."""
    signal = PeerSignal("a", "b", STUDIES, peer=0.06, supervised=0.18, symmetric_kl=0.03, correlation=0.9)
    assert signal.ratio() == pytest.approx(1 / 3)

    empty = PeerSignal("a", "b", 0, peer=0.0, supervised=0.0, symmetric_kl=0.0, correlation=0.0)
    assert empty.ratio() == 0.0


def test_line() -> None:
    """Carries both terms and the ratio, because either term alone has no scale to be read on."""
    signal = PeerSignal("a", "b", STUDIES, peer=0.06, supervised=0.18, symmetric_kl=0.03, correlation=0.9)
    line = signal.line()
    assert "a vs b" in line
    assert "peer=0.0600" in line
    assert "supervised=0.1800" in line
    assert "ratio=0.333" in line


def test_between() -> None:
    """Joins on shared studies, reads zero for a member against itself, and refuses disjoint draws."""
    names = _names()
    one = _reading(names, seed=1)
    comparison = PeerComparison({"one": one, "twin": one, "other": _reading(names, seed=2)})

    itself = comparison.between("one", "twin")
    assert itself.peer < IDENTICAL_TOLERANCE
    assert itself.symmetric_kl < IDENTICAL_TOLERANCE
    assert itself.correlation == pytest.approx(1.0)

    apart = comparison.between("one", "other")
    assert apart.peer > itself.peer
    assert apart.studies == STUDIES
    assert apart.supervised > 0.0

    disjoint = PeerComparison({"here": _reading(_names(), 3), "elsewhere": _reading(["z0", "z1"], 4)})
    with pytest.raises(ValueError, match="share no studies"):
        disjoint.between("here", "elsewhere")


def test_across() -> None:
    """Reports every pair once, so a cohort is judged on its weakest disagreement and not its best."""
    names = _names()
    comparison = PeerComparison({stem: _reading(names, seed) for seed, stem in enumerate(["a", "b", "c"])})

    signals = comparison.across()
    assert [(signal.left, signal.right) for signal in signals] == [("a", "b"), ("a", "c"), ("b", "c")]
    assert all(signal.ratio() > 0.0 for signal in signals)
