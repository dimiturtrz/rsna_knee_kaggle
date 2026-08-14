"""The published-member oracle, on checkpoints written by hand so the right answer is known."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kneemri.members import Member
from kneemri.oof_oracle import OutOfFoldOracle

TARGET_COUNT = 12
SEEDS = (2026, 7717)
FOLDS = (
    {"a", "b"},
    {
        "c",
    },
)


def _checkpoint(path: Path, seed: int, identifiers: list[str], *, crop_mm: float = 130.0) -> Path:
    """A checkpoint carrying only what this oracle reads: the oof block, the config and the slots."""
    rows = [[float(index + position) / 100.0 for position in range(TARGET_COUNT)] for index in range(len(identifiers))]
    torch.save(
        {
            "oof": {"ids": identifiers, "pred": rows},
            "config": {"seed": seed, "crop_mm": crop_mm, "band": "0.20,0.80", "slots": ["SAG_FLUID_FS"]},
            "slots": ["SAG_FLUID_FS"],
        },
        path,
    )
    return path


def test_recorded(tmp_path: Path):
    """The author's own identifiers and probabilities, read back in their written order."""
    path = _checkpoint(tmp_path / "m.pt", 2026, ["b", "a"])

    identifiers, values = OutOfFoldOracle().recorded(path)

    assert identifiers == ["b", "a"], "their order, not a sorted one — the rows are aligned to it"
    assert values.shape == (2, TARGET_COUNT)


def test_corpus(tmp_path: Path):
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

    names, per_seed = OutOfFoldOracle().corpus(published)

    assert sorted(names) == ["a", "b", "c"], "the folds partition the corpus and every study appears once"
    assert len(per_seed) == len(SEEDS)
    assert all(array.shape == (3, TARGET_COUNT) for array in per_seed)
    first, second = (names.index("a"), names.index("b"))
    # seed 2 wrote b first; row 'a' must still carry the values it wrote FOR a.
    assert per_seed[1][first, 0] != per_seed[1][second, 0]
    assert per_seed[1][names.index("a"), 0] == per_seed[0][names.index("b"), 0], "index 1 of its own list"


def test_agrees(tmp_path: Path):
    """Silence when the recorded reading matches ours, and a named difference when it does not."""
    oracle = OutOfFoldOracle()
    member = Member.from_checkpoint(_checkpoint(tmp_path / "same.pt", 2026, ["a"]))

    assert oracle.agrees(member, _checkpoint(tmp_path / "s.pt", 2026, ["a"])) == []

    differing = oracle.agrees(member, _checkpoint(tmp_path / "d.pt", 2026, ["a"], crop_mm=99.0))
    assert len(differing) == 1
    assert "crop_mm" in differing[0]


def test_covariates():
    """Per-study properties that cost nothing, aligned to the identifiers they describe."""
    series = pd.DataFrame({"StudyInstanceUID": ["a", "a", "b"], "SeriesInstanceUID": ["1", "2", "3"]})

    covariates = OutOfFoldOracle().covariates(["b", "a", "missing"], series)

    assert list(covariates["series_count"]) == [1.0, 2.0, 0.0], "aligned to the given order, absent means zero"


def test_report():
    """The residual, its concentration, and the worst studies named so they can be looked at."""
    identifiers = ["quiet", "loud"]
    ours = np.zeros((2, TARGET_COUNT))
    theirs = np.zeros((2, TARGET_COUNT))
    theirs[1, 0] = 0.4
    covariates = {"series_count": np.array([1.0, 9.0])}

    lines = OutOfFoldOracle().report(identifiers, ours, theirs, covariates)
    text = "\n".join(lines)

    assert "max|diff| 0.400000" in text
    assert "studies above 0.01: 1 of 2" in text
    assert "corr(residual, series_count) = +1.000" in text, "a defect that tracks a covariate says so"
    assert text.index("loud") < text.index("quiet"), "worst first, because that is what gets opened"
