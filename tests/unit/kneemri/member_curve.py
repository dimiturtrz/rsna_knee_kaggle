"""The member curve's arithmetic, on constructed predictions whose right answer is known by hand."""

import numpy as np

from kneemri.member_curve import MemberCurve
from kneemri.model import TARGETS
from kneemri.scores import LabelScores

STUDIES = 40
SEED = 2026


def _truth() -> np.ndarray:
    """Half the studies positive on every label."""
    column = np.array([1.0] * (STUDIES // 2) + [0.0] * (STUDIES // 2), dtype=np.float32)
    return np.repeat(column[:, None], len(TARGETS), axis=1)


def _member(strength: float, seed: int) -> np.ndarray:
    """A member that ranks the positives above the negatives with the given reliability."""
    generator = np.random.default_rng(seed)
    signal = _truth()[:, 0].astype(np.float64) * strength
    return np.repeat((signal + generator.normal(0.0, 1.0, STUDIES))[:, None], len(TARGETS), axis=1)


def test_sizes():
    """Three members give three singletons, three pairs and one triple."""
    members = [_member(1.0, seed) for seed in (1, 2, 3)]
    by_size = MemberCurve(LabelScores()).sizes(members, _truth())
    assert [len(by_size[size]) for size in (1, 2, 3)] == [3, 3, 1]


def test_sizes_shows_independent_members_helping():
    """Averaging ranks of independent noisy members beats any one of them.

    This is the property the whole experiment exists to measure, so a curve that could not detect it
    would be measuring nothing. The members here are independent by construction; real ones share
    training data and are not, which is exactly why the real curve has to be run rather than assumed.
    """
    members = [_member(1.0, seed) for seed in (1, 2, 3)]
    by_size = MemberCurve(LabelScores()).sizes(members, _truth())
    assert np.mean(by_size[3]) > np.mean(by_size[1])


def test_sizes_is_flat_for_duplicated_members():
    """A member added twice adds nothing — the guard against reading ensembling into a copy."""
    duplicated = [_member(1.0, 1)] * 3
    by_size = MemberCurve(LabelScores()).sizes(duplicated, _truth())
    assert np.mean(by_size[3]) == np.mean(by_size[1])


def test_report():
    """One line per size, carrying the mean, the range and the step from the previous size."""
    lines = MemberCurve(LabelScores()).report({1: [0.80, 0.82], 2: [0.85]})
    assert lines[0].startswith("1 member(s): mean 0.8100")
    assert "+0.0400 over 1" in lines[1]


def test_subsets():
    """Every subset of one size, carrying the INDICES that identify it and not just its score."""
    curve = MemberCurve(LabelScores())
    truth = _truth()
    members = [_member(2.0, SEED), _member(0.5, SEED + 1)]

    pairs = curve.subsets(members, truth, size=1)
    assert [subset for _, subset in pairs] == [(0,), (1,)], "one entry per member, each naming itself"
    assert all(np.isfinite(score) for score, _ in pairs)
    assert [subset for _, subset in curve.subsets(members, truth, size=2)] == [(0, 1)]


def test_best():
    """The winning subset is NAMED at every size, which is what a submission is built from.

    `report` said five members reached 0.8601 and never said WHICH five, so the best number this
    analysis produced could not be turned into an entry.
    """
    curve = MemberCurve(LabelScores())
    truth = _truth()
    members = [_member(2.0, SEED), _member(0.2, SEED + 1)]

    lines = curve.best(members, truth, ["strong.pt", "weak.pt"])
    assert len(lines) == 2
    assert "strong.pt" in lines[0] and "weak.pt" not in lines[0], "the best single member is the strong one"
    assert "strong.pt" in lines[1] and "weak.pt" in lines[1], "the only pair is both of them"


def test_oracle():
    """The ceiling is at least the plain average, and strictly above it when members differ per label.

    Constructed so member A is perfect on the first label and useless on the second while member B is
    the reverse. Averaging them cannot beat either on the label it owns, so the per-label ceiling must
    exceed the average - which is exactly the case the measurement exists to detect.
    """
    truth = np.zeros((6, len(TARGETS)), dtype=np.float32)
    truth[3:, 0] = 1.0
    truth[3:, 1] = 1.0
    first = np.tile(np.array([[0.1], [0.2], [0.3], [0.7], [0.8], [0.9]]), (1, len(TARGETS)))
    second = first.copy()
    second[:, 0] = first[::-1, 0]  # B is inverted on label 0
    first[:, 1] = second[::-1, 1].copy()  # A is inverted on label 1

    ceiling = MemberCurve(LabelScores()).oracle([first, second], truth)

    assert ceiling["oracle"] >= ceiling["average"], "a ceiling below the thing it bounds is not a ceiling"
    assert ceiling["oracle"] > ceiling["average"], "members that differ per label must show headroom"
    assert ceiling["average"] >= 0.0


def test_paired():
    """Reports one block per member against the first, and stars only intervals that exclude zero."""
    generator = np.random.default_rng(11)
    truth = np.zeros((120, len(TARGETS)), dtype=np.float32)
    truth[generator.choice(120, 40, replace=False), :] = 1.0
    signal = truth + generator.normal(0, 1.0, truth.shape)
    control = (signal + generator.normal(0, 0.05, truth.shape)).astype(np.float64)
    # NOT a rescaling of `signal`: AUC is invariant to any increasing map, so `signal * 3` ranks
    # identically and would be no better at all. Extra TRUTH is what actually separates.
    better = (signal + truth * 2.0).astype(np.float64)

    lines = MemberCurve(LabelScores()).paired([control, better], truth, ["control.pt", "better.pt"])
    text = "\n".join(lines)

    assert "paired deltas, better.pt against control.pt:" in text
    assert "*" in text, "a member built to be strictly better must clear the interval somewhere"
