"""The member curve's arithmetic, on constructed predictions whose right answer is known by hand."""

from pathlib import Path

import numpy as np
import torch

from kneemri.holdout_predictions import GROUP
from kneemri.member_curve import MemberCurve, Reading
from kneemri.model import TARGETS
from kneemri.scores import LabelScores
from kneemri.slots import SLOTS

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


def _checkpoint(tmp_path: Path, name: str, config: dict) -> Path:
    path = tmp_path / f"{name}.pt"
    torch.save({"model": {}, "config": config}, path)
    return path


def test_of(tmp_path: Path):
    """A reading is taken from the checkpoint, with fallbacks only where a field predates recording."""
    full = _checkpoint(tmp_path, "full", {"img": 448, "train_size": 392, "slices": 12, "whole_stack": True,
                                          "variant": "small", "slots": [slot.name for slot in SLOTS]})
    reading = Reading.of(full)
    assert (reading.image_size, reading.read_size, reading.whole_stack) == (448, 392, True)

    bare = _checkpoint(tmp_path, "bare", {"img": 336})
    legacy = Reading.of(bare)
    assert legacy.read_size == 336, "no train_size means the cache's own size"
    assert not legacy.whole_stack, "a member predating the flag was windowed"
    assert legacy.slots == len(SLOTS)


def test_cache_key(tmp_path: Path):
    """Members sharing a cache key share a load; a different resolution is a different cache."""
    small = Reading.of(_checkpoint(tmp_path, "a", {"img": 336, "slices": 12}))
    large = Reading.of(_checkpoint(tmp_path, "b", {"img": 448, "slices": 12}))
    twin = Reading.of(_checkpoint(tmp_path, "c", {"img": 336, "slices": 12}))
    assert small.cache_key == twin.cache_key
    assert small.cache_key != large.cache_key


def test_width(tmp_path: Path):
    """A whole-stack member's window is its whole stack; a windowed member's is the three-slice group."""
    whole = Reading.of(_checkpoint(tmp_path, "w", {"img": 336, "slices": 12, "whole_stack": True}))
    windowed = Reading.of(_checkpoint(tmp_path, "n", {"img": 336, "slices": 12}))
    assert whole.width == 12
    assert windowed.width == GROUP


def test_window(tmp_path: Path):
    """One centred full-width window for a whole-stack member, whatever --overlap asks for.

    Overlap is a choice about windowed members. A whole-stack member has no windows to overlap: its
    slot embedding is the maximum over the whole stack, and scoring it three slices at a time measures
    a different function - which is what this CLI did to every whole-stack member it ever ranked.
    """
    whole = Reading.of(_checkpoint(tmp_path, "w2", {"img": 336, "slices": 12, "whole_stack": True}))
    window = whole.window(overlap=True)
    assert window.starts == [0] and window.width == 12

    windowed = Reading.of(_checkpoint(tmp_path, "n2", {"img": 336, "slices": 12}))
    assert len(windowed.window(overlap=True).starts) == 12 - GROUP + 1
    assert windowed.window(overlap=False).starts == [0, 3, 6, 9]
