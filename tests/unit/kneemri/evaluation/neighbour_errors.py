"""A neighbour vote over a hand-built feature space, so every cell's right answer is known before the run.

The corpus is two well-separated clusters whose labels are constant inside each cluster, with three cells
flipped. A detector that works must score exactly those three lowest — "did it find the planted flip" is a
question with an answer, unlike a separation AUC over real features, which is a number to admire.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.findings import STUDY, TARGETS
from kneemri.evaluation.label_errors import VERDICT_SUFFIX, ReaderTable
from kneemri.evaluation.neighbour_errors import MAJORITY, CombinedRanking, NeighbourVote, TrunkView

STUDIES = 40
PLANTED = (3, 21, 34)
"""Rows whose verdict contradicts every neighbour they have — the errors the vote must rank worst."""

FINDING = TARGETS[0]
NEIGHBOURS = 5


def _clustered(dimension: int = 8) -> np.ndarray:
    """Two tight clusters, far apart, one per half of the corpus."""
    generator = np.random.default_rng(0)
    centres = np.zeros((2, dimension), dtype=np.float32)
    centres[0, 0] = 10.0
    centres[1, 1] = 10.0
    half = STUDIES // 2
    noise = generator.normal(scale=0.01, size=(STUDIES, dimension)).astype(np.float32)
    return np.concatenate([np.tile(centres[0], (half, 1)), np.tile(centres[1], (STUDIES - half, 1))]) + noise


def _labels() -> np.ndarray:
    """Cluster one is positive and cluster two is negative, then three cells are flipped."""
    given = np.zeros(STUDIES, dtype=bool)
    given[: STUDIES // 2] = True
    given[list(PLANTED)] = ~given[list(PLANTED)]
    return given


def _reader() -> ReaderTable:
    """One finding read on every study, every other finding unread."""
    given = _labels()
    verdicts = ["YES" if value else "NO" for value in given]
    table = pd.DataFrame({STUDY: [f"s{index}" for index in range(STUDIES)], f"{FINDING}{VERDICT_SUFFIX}": verdicts})
    return ReaderTable.of(table, [f"s{index}" for index in range(STUDIES)])


def _view() -> TrunkView:
    """The clustered corpus as a single trunk view, normalised the way `TrunkView.of` normalises."""
    vectors = _clustered()
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return TrunkView([f"s{index}" for index in range(STUDIES)], unit.astype(np.float32))


def _written(directory: Path, key: str, slots: int = 2, slices: int = 3, dimension: int = 4) -> None:
    """A three-artefact embedding cache on disk, with the second slot absent for the first study."""
    features = np.zeros((STUDIES, slots, slices, dimension), dtype=np.float16)
    features[:, 0] = 2.0
    features[:, 1] = 6.0
    present = np.ones((STUDIES, slots), dtype=bool)
    present[0, 1] = False
    np.save(directory / f"features_{key}.npy", features)
    np.save(directory / f"features_mask_{key}.npy", present)
    (directory / f"features_index_{key}.json").write_text(json.dumps([f"s{index}" for index in range(STUDIES)]))


def test_trunk_view_of(tmp_path: Path):
    """The cache loads in its stored order, pooled over present slots only and put on the unit sphere."""
    _written(tmp_path, "toy")
    view = TrunkView.of(tmp_path, "toy")
    assert view.identifiers == [f"s{index}" for index in range(STUDIES)]
    assert view.unit.shape == (STUDIES, 4)
    assert np.allclose(np.linalg.norm(view.unit, axis=1), 1.0)


def test_trunk_view_of_ignores_absent_slots(tmp_path: Path):
    """The study missing its second slot is pooled from the first alone, not averaged toward zero.

    Both slots are constant, so the direction is identical either way and only the magnitude would differ —
    which is exactly why the check is on the pooled value before normalisation rather than after it.
    """
    _written(tmp_path, "toy")
    pooled = TrunkView._pooled(np.load(tmp_path / "features_toy.npy"), np.load(tmp_path / "features_mask_toy.npy"))
    assert pooled[0] == pytest.approx(2.0)
    assert pooled[1] == pytest.approx(4.0)


def test_neighbour_vote_agreement():
    """The three planted flips score lowest, and every unread finding stays NaN."""
    vote = NeighbourVote((_view(),), _reader(), NEIGHBOURS)
    agreement = vote.agreement()
    column = agreement[:, TARGETS.index(FINDING)]
    assert np.isnan(agreement[:, TARGETS.index(TARGETS[1])]).all()
    assert set(np.argsort(column)[: len(PLANTED)]) == set(PLANTED)
    assert column[list(PLANTED)].max() < column[[index for index in range(STUDIES) if index not in PLANTED]].min()


def test_neighbour_vote_agreement_averages_views():
    """Two identical views give the single view's answer — the aggregation is a mean, not a sum."""
    view = _view()
    one = NeighbourVote((view,), _reader(), NEIGHBOURS).agreement()
    two = NeighbourVote((view, view), _reader(), NEIGHBOURS).agreement()
    assert np.allclose(one, two, equal_nan=True)


def test_trunk_view_loaded(tmp_path: Path):
    """A named cache that is not on the machine is skipped, and no cache at all is a refusal."""
    _written(tmp_path, "toy")
    views = TrunkView.loaded(tmp_path, ("toy", "absent"))
    assert len(views) == 1
    assert views[0].unit.shape == (STUDIES, 4)
    with pytest.raises(FileNotFoundError):
        TrunkView.loaded(tmp_path, ("absent",))


def test_trunk_view_aligned():
    """Views sharing a study order give it back; views that disagree refuse rather than join."""
    view = _view()
    assert TrunkView.aligned((view, view)) == view.identifiers
    shuffled = TrunkView(list(reversed(view.identifiers)), view.unit)
    with pytest.raises(ValueError, match="do not share a study order"):
        TrunkView.aligned((view, shuffled))


def test_combined_ranking_onto():
    """Cells follow their study identifier, and a target study the source lacks stays unread."""
    scores = np.array([[1.0, 2.0], [3.0, 4.0]])
    placed = CombinedRanking.onto(scores, ["a", "b"], ["b", "missing", "a"])
    assert np.allclose(placed[0], [3.0, 4.0])
    assert np.allclose(placed[2], [1.0, 2.0])
    assert np.isnan(placed[1]).all()


def test_combined_ranking_quality():
    """Scale cannot outvote order, unread cells stay unread, and one part alone is that part's ranking."""
    wide = np.array([[0.0, 100.0, 200.0, np.nan]])
    narrow = np.array([[0.02, 0.01, 0.03, np.nan]])
    alone = CombinedRanking((wide,)).quality()
    assert np.allclose(alone[0, :3], [0.0, 0.5, 1.0])
    assert np.isnan(alone[0, 3])
    combined = CombinedRanking((wide, narrow)).quality()
    assert np.isnan(combined[0, 3])
    assert combined[0, 0] == pytest.approx(0.25)
    assert combined[0, 1] == pytest.approx(0.25)
    assert combined[0, 2] == pytest.approx(1.0)


def test_neighbour_vote_dissent():
    """Exactly the planted cells fall under the majority line, and no unread cell is ever flagged."""
    vote = NeighbourVote((_view(),), _reader(), NEIGHBOURS)
    dissent = vote.dissent()
    assert set(np.flatnonzero(dissent[:, TARGETS.index(FINDING)])) == set(PLANTED)
    assert not dissent[:, TARGETS.index(TARGETS[1])].any()
    assert vote.agreement()[list(PLANTED), TARGETS.index(FINDING)].max() < MAJORITY
