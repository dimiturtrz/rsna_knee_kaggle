"""The whole-corpus ruler, on checkpoints written by hand so the right answer is known."""

from pathlib import Path

import numpy as np
import torch

from core.data.corpus.slots import SLOTS
from kneemri.evaluation.published_ruler import PublishedFold, PublishedRuler

TARGET_COUNT = 12
CANONICAL = [slot.name for slot in SLOTS]
SEEDS = (2026, 7717)


def _checkpoint(path: Path, seed: int, identifiers: list[str]) -> Path:
    """A checkpoint carrying only what the ruler reads: the oof block and the seed."""
    rows = [[float(index + position) / 100.0 for position in range(TARGET_COUNT)] for index in range(len(identifiers))]
    torch.save(
        {
            "oof": {"ids": identifiers, "pred": rows},
            "config": {"seed": seed, "crop_mm": 130.0, "band": "0.20,0.80", "slots": CANONICAL},
            "slots": CANONICAL,
        },
        path,
    )
    return path


def test_of(tmp_path: Path):
    """The seed and the author's own identifiers, read back in their written order."""
    entry = PublishedFold.of(_checkpoint(tmp_path / "m.pt", 2026, ["b", "a"]))

    assert entry.seed == 2026
    assert entry.identifiers == ["b", "a"], "their order, not a sorted one — the rows are aligned to it"
    assert entry.values.shape == (2, TARGET_COUNT)


def test_fold(tmp_path: Path):
    """The fold key is order-independent, which is the only thing fold-mates are promised to share."""
    first = PublishedFold.of(_checkpoint(tmp_path / "a.pt", SEEDS[0], ["a", "b"]))
    second = PublishedFold.of(_checkpoint(tmp_path / "b.pt", SEEDS[1], ["b", "a"]))
    other = PublishedFold.of(_checkpoint(tmp_path / "c.pt", SEEDS[0], ["c"]))

    assert first.fold() == second.fold(), "same studies written the other way round is the SAME fold"
    assert first.fold() != other.fold()


def test_aligned_to(tmp_path: Path):
    """Rows follow the order asked for, carrying each study's own values with it."""
    entry = PublishedFold.of(_checkpoint(tmp_path / "m.pt", 2026, ["b", "a"]))

    aligned = entry.aligned_to(["a", "b"])

    assert np.array_equal(aligned[0], entry.values[1]), "row 'a' keeps the values written FOR a"
    assert np.array_equal(aligned[1], entry.values[0])


def test_per_seed(tmp_path: Path):
    """Folds stacked into one full-corpus array per seed, with each fold's rows REORDERED to match.

    The reordering is the part worth testing: two members of one fold hold the same studies and nothing
    promises they wrote them in the same order, so a naive concatenation would misalign one seed against
    another and produce a residual no amount of staring at the reader would explain.
    """
    published = tmp_path / "published"
    published.mkdir()
    _checkpoint(published / "f1_s1.pt", SEEDS[0], ["a", "b"])
    _checkpoint(published / "f1_s2.pt", SEEDS[1], ["b", "a"])  # SAME fold, written the other way round
    _checkpoint(published / "f2_s1.pt", SEEDS[0], ["c"])
    _checkpoint(published / "f2_s2.pt", SEEDS[1], ["c"])

    names, per_seed = PublishedRuler(published).per_seed()

    assert sorted(names) == ["a", "b", "c"], "the folds partition the corpus and every study appears once"
    assert len(per_seed) == len(SEEDS)
    assert all(array.shape == (3, TARGET_COUNT) for array in per_seed)
    first, second = (names.index("a"), names.index("b"))
    # seed 2 wrote b first; row 'a' must still carry the values it wrote FOR a.
    assert per_seed[1][first, 0] != per_seed[1][second, 0]
    assert per_seed[1][names.index("a"), 0] == per_seed[0][names.index("b"), 0], "index 1 of its own list"


def test_per_seed_skips_a_fold_that_is_missing_a_seed(tmp_path: Path):
    """A fold present for one seed of two is dropped WHOLE, not padded.

    The arrays are one block per seed, so a fold with no row to give the second seed would make the two
    columns cover different populations — a ruler whose two halves measure different corpora, which is
    worse than a shorter one.
    """
    published = tmp_path / "published"
    published.mkdir()
    _checkpoint(published / "f1_s1.pt", SEEDS[0], ["a", "b"])
    _checkpoint(published / "f1_s2.pt", SEEDS[1], ["a", "b"])
    _checkpoint(published / "f2_s1.pt", SEEDS[0], ["c"])  # no fold-2 member for the second seed

    names, per_seed = PublishedRuler(published).per_seed()

    assert sorted(names) == ["a", "b"], "the half-covered fold contributes nothing"
    assert all(array.shape == (2, TARGET_COUNT) for array in per_seed)
