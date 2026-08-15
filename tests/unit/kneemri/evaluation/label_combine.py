"""Per-label combining, on members built so the right answer is known before it is measured."""

import numpy as np

from kneemri.evaluation.label_combine import LabelCombine
from kneemri.evaluation.scores import LabelScores
from kneemri.models.model import TARGETS

STUDIES = 120
SPECIALISTS = 2
"""One member strong on the first finding and weak on the second; the other the exact reverse."""


def _truth(rng: np.random.Generator) -> np.ndarray:
    """Graded targets, half of them positive on every finding."""
    positives = rng.permutation(STUDIES) < STUDIES // 2
    return np.tile(np.where(positives, 1.0, 0.0)[:, None], (1, len(TARGETS))).astype(np.float32)


def _specialists(truth: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Two members that each see ONE finding, so per-label selection has something real to find.

    Both are noise on every other finding, which is what makes the flat average of the two worse per
    label than the one member that can actually see it.
    """
    members = []
    for expert in range(SPECIALISTS):
        member = rng.random((STUDIES, len(TARGETS)))
        member[:, expert] = truth[:, expert] * 0.5 + rng.random(STUDIES) * 0.1
        members.append(member)
    return members


def test_ranking():
    """Each finding's members ordered best-first, and an undefined AUC ordered last rather than loose."""
    rng = np.random.default_rng(0)
    truth = _truth(rng)
    combine = LabelCombine(LabelScores())

    ranking = combine.ranking(_specialists(truth, rng), truth)

    assert ranking[TARGETS[0]][0] == 0, "the member that can see the first finding ranks first for it"
    assert ranking[TARGETS[1]][0] == 1

    single = np.zeros((STUDIES, len(TARGETS)), dtype=np.float32)
    blind = combine.ranking([rng.random((STUDIES, len(TARGETS))), rng.random((STUDIES, len(TARGETS)))], single)
    assert len(blind[TARGETS[0]]) == SPECIALISTS, "an all-negative label is undefined, not an error"


def test_applied():
    """Top-k per label, and k = every member reproducing the flat rank-average exactly."""
    rng = np.random.default_rng(1)
    truth = _truth(rng)
    combine = LabelCombine(LabelScores())
    members = _specialists(truth, rng)
    ranking = combine.ranking(members, truth)
    scores = LabelScores()

    from kneemri.inference.ensemble import Ensemble  # noqa: PLC0415 - the control this method must reproduce

    everyone = combine.applied(members, ranking, SPECIALISTS)
    assert scores.mean(everyone, truth) == scores.mean(Ensemble.combine(members), truth)

    chosen = scores.per_label(combine.applied(members, ranking, 1), truth)
    flat = scores.per_label(everyone, truth)
    assert chosen[TARGETS[0]] > flat[TARGETS[0]], "the specialist alone beats itself diluted by a coin"


def test_halves():
    """Two (fit, report) splits that partition the studies and swap roles."""
    combine = LabelCombine(LabelScores())

    splits = combine.halves(STUDIES, seed=7)

    assert len(splits) == SPECIALISTS
    first, second = splits
    assert sorted(np.concatenate(first).tolist()) == list(range(STUDIES)), "a partition, not a sample"
    assert np.array_equal(first[0], second[1]), "each half fits once and reports once"
    assert np.array_equal(first[1], second[0])
    assert not set(first[0].tolist()) & set(first[1].tolist()), "nothing is fitted and reported on"


def test_honest():
    """The estimate is computed where the choice was NOT made — the point of the whole module."""
    rng = np.random.default_rng(2)
    truth = _truth(rng)
    combine = LabelCombine(LabelScores())
    members = _specialists(truth, rng)

    assert combine.honest(members, truth, 1, seed=3) > combine.honest(members, truth, SPECIALISTS, seed=3), (
        "a real per-label specialism survives being chosen on studies it is not scored on"
    )

    coins = [rng.random((STUDIES, len(TARGETS))) for _ in range(SPECIALISTS)]
    inside = LabelScores().mean(combine.applied(coins, combine.ranking(coins, truth), 1), truth)
    assert inside > combine.honest(coins, truth, 1, seed=3), (
        "members carrying no signal are still SELECTABLE in-sample, and that optimism is what the "
        "split-half estimate exists to strip"
    )


def test_curve():
    """One line per k, the flat rank-average named as the control it is."""
    rng = np.random.default_rng(4)
    truth = _truth(rng)
    members = _specialists(truth, rng)

    lines = LabelCombine(LabelScores()).curve(members, truth, seed=5)

    assert len(lines) == SPECIALISTS
    assert "in-sample" in lines[0] and "split-half" in lines[0]
    assert "(= the flat rank-average)" in lines[-1], "the matched control is the last row, under one ruler"
    assert "(= the flat rank-average)" not in lines[0]
