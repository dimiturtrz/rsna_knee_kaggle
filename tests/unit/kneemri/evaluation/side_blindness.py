"""The discordant contrast, on predictions this test constructs to have a known side-blindness."""

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.scores import LabelScores
from kneemri.evaluation.side_blindness import CHANCE, COMPARTMENT_PAIRS, Discordant, SideBlindness

STUDIES = 240
RESAMPLES = 60


def _columns() -> dict[str, int]:
    return {target: index for index, target in enumerate(TARGETS)}


def _truth(columns: dict[str, int]) -> np.ndarray:
    """Half the studies positive on the medial side only, half on the lateral side only.

    INTERLEAVED, not split into blocks. A blocked layout makes row index a perfect proxy for the side,
    so any ramp over rows — including the deliberately side-blind one below — separates the groups
    perfectly and the fixture proves the opposite of what it was written to prove.
    """
    truth = np.zeros((STUDIES, len(TARGETS)), dtype=np.float32)
    for medial_name, lateral_name in COMPARTMENT_PAIRS:
        truth[0::2, columns[medial_name]] = 1.0
        truth[1::2, columns[lateral_name]] = 1.0
    return truth


def test_of():
    """The two groups are EXCLUSIVE: a study positive on both sides belongs to neither."""
    columns = _columns()
    truth = _truth(columns)
    pair = COMPARTMENT_PAIRS[0]
    truth[0, columns[pair[1]]] = 1.0  # study 0 is now positive on both sides

    groups = Discordant.of(pair, truth, columns)

    assert not groups.lateral_only[0] and not groups.medial_only[0], "a both-sided study is discordant on neither"
    assert not (groups.lateral_only & groups.medial_only).any()
    assert int(groups.medial_only.sum()) == STUDIES // 2 - 1


def test_contrast():
    """A column that copies one presence signal into both sides scores exactly chance, in both directions."""
    columns = _columns()
    truth = _truth(columns)
    pair = COMPARTMENT_PAIRS[0]
    groups = Discordant.of(pair, truth, columns)
    blind = np.zeros((STUDIES, len(TARGETS)), dtype=np.float64)
    blind[:, columns[pair[0]]] = np.linspace(0.0, 1.0, STUDIES)
    blind[:, columns[pair[1]]] = blind[:, columns[pair[0]]]

    scores = LabelScores()
    # A ramp shared by both columns is not a constant, so a chance reading here is the contrast dividing
    # the side out rather than an AUC over ties.
    for label in pair:
        raw = groups.contrast(label, blind, columns, scores)
        assert abs(groups.oriented(label, raw) - CHANCE) < 0.08

    seeing = blind.copy()
    seeing[:, columns[pair[1]]] = groups.lateral_only.astype(np.float64)
    assert groups.contrast(pair[1], seeing, columns, scores) == 1.0


def test_interval_is_pinned():
    """Bit-equal to the hand-written 2.5/97.5 tail, on noise that leaves the contrast a real spread."""
    columns = _columns()
    truth = _truth(columns)
    predictions = np.random.default_rng(3).random((STUDIES, len(columns)))
    pair = COMPARTMENT_PAIRS[0]
    groups = Discordant.of(pair, truth, columns)

    interval = SideBlindness(LabelScores(), resamples=RESAMPLES)._interval(groups, pair[1], predictions, columns)

    assert interval == (0.5033298611111111, 0.6393246527777778)


def test_oriented():
    """The medial column runs backwards, so its raw contrast is flipped and the lateral one is not."""
    groups = Discordant.of(COMPARTMENT_PAIRS[0], _truth(_columns()), _columns())

    assert groups.oriented(COMPARTMENT_PAIRS[0][0], 0.25) == pytest.approx(0.75)
    assert groups.oriented(COMPARTMENT_PAIRS[0][1], 0.25) == pytest.approx(0.25)


def test_lines():
    """A pair the model resolves is named as such; one it cannot is named INDISTINGUISHABLE."""
    columns = _columns()
    truth = _truth(columns)
    predictions = np.tile(np.linspace(0.0, 1.0, STUDIES)[:, None], (1, len(TARGETS)))
    resolved = COMPARTMENT_PAIRS[0]
    predictions[:, columns[resolved[1]]] = Discordant.of(resolved, truth, columns).lateral_only.astype(np.float64)

    report = SideBlindness(LabelScores(), resamples=RESAMPLES).lines(["a", "b"], predictions, truth, columns)

    assert report[0].startswith("discordant contrast over 240 studies, 2 members combined")
    resolves = [line for line in report if "resolves the side" in line]
    assert any(resolved[1] in line for line in resolves), "the perfectly side-aware column must read as resolving"
    assert any("INDISTINGUISHABLE FROM SIDE-BLIND" in line for line in report), (
        "a column shared with its partner must read as blind"
    )


def test_lines_names_an_unreadable_pair():
    """No discordant studies is reported, never silently skipped — an empty contrast is not a 0.50."""
    columns = _columns()
    truth = np.zeros((STUDIES, len(TARGETS)), dtype=np.float32)
    truth[:, columns[COMPARTMENT_PAIRS[0][0]]] = 1.0
    truth[:, columns[COMPARTMENT_PAIRS[0][1]]] = 1.0

    blank = np.zeros_like(truth, dtype=np.float64)
    report = SideBlindness(LabelScores(), resamples=RESAMPLES).lines(["a"], blank, truth, columns)

    assert any("UNREADABLE" in line for line in report)
