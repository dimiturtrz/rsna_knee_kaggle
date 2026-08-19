"""The published-member oracle, on checkpoints written by hand so the right answer is known."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kneemri.corpus.slots import SLOTS
from kneemri.evaluation.oof_oracle import OutOfFoldOracle, OutOfFoldOracleReport, PublishedCheckpoint
from kneemri.inference.members import Member

from ..conftest import write_series

TARGET_COUNT = 12
CANONICAL = [slot.name for slot in SLOTS]
"""The published members use the canonical six, and `agrees` compares their order against OURS."""


def _checkpoint(path: Path, seed: int, identifiers: list[str], *, crop_mm: float = 130.0) -> Path:
    """A checkpoint carrying only what this oracle reads: the oof block, the config and the slots."""
    rows = [[float(index + position) / 100.0 for position in range(TARGET_COUNT)] for index in range(len(identifiers))]
    torch.save(
        {
            "oof": {"ids": identifiers, "pred": rows},
            "config": {"seed": seed, "crop_mm": crop_mm, "band": "0.20,0.80", "slots": CANONICAL},
            "slots": CANONICAL,
        },
        path,
    )
    return path


def test_recorded(tmp_path: Path):
    """The author's own identifiers and probabilities, read back in their written order."""
    path = _checkpoint(tmp_path / "m.pt", 2026, ["b", "a"])

    identifiers, values = PublishedCheckpoint.read(path).recorded()

    assert identifiers == ["b", "a"], "their order, not a sorted one — the rows are aligned to it"
    assert values.shape == (2, TARGET_COUNT)


def test_agrees(tmp_path: Path):
    """Silence when the recorded reading matches ours, and a named difference when it does not."""
    member = Member.from_checkpoint(_checkpoint(tmp_path / "same.pt", 2026, ["a"]))

    assert PublishedCheckpoint.read(_checkpoint(tmp_path / "s.pt", 2026, ["a"])).agrees(member) == []

    differing = PublishedCheckpoint.read(_checkpoint(tmp_path / "d.pt", 2026, ["a"], crop_mm=99.0)).agrees(member)
    assert len(differing) == 1
    assert "crop_mm" in differing[0]

    # AGAINST THE MEMBER, not against our module constants. This compared with CROP_MM and SLICE_BAND
    # and never touched the `Member` it was handed, which asked the same question only while nothing of
    # ours could differ from the default — so a member fitted on the 80 mm cache was reported as
    # differing from the pipeline that was about to serve it correctly.
    tighter = _checkpoint(tmp_path / "t.pt", 2026, ["a"], crop_mm=80.0)
    assert PublishedCheckpoint.read(tighter).agrees(Member.from_checkpoint(tighter)) == [], (
        "a member matches its own reading"
    )

    # A field their file states and our accessor DROPS is the failure this really watches for: the
    # member would then be served our default and the difference charged to the reader as a residual.
    silent = torch.load(tmp_path / "same.pt", map_location="cpu", weights_only=False)
    del silent["config"]["band"]
    torch.save(silent, tmp_path / "silent.pt")
    unstated = PublishedCheckpoint.read(tmp_path / "silent.pt").agrees(member)
    assert len(unstated) == 1
    assert "no band" in unstated[0], "an absent field is a difference from what we will serve, not a default"


def test_read(tmp_path: Path):
    """One deserialisation answers both questions — the point of holding the payload rather than the path."""
    path = _checkpoint(tmp_path / "m.pt", 2026, ["a"])

    published = PublishedCheckpoint.read(path)

    assert set(published.saved) == {"oof", "config", "slots"}
    assert published.recorded()[0] == ["a"], "the same object answers the oof question"
    assert published.agrees(Member.from_checkpoint(path)) == [], "and the reading question, without re-reading"


def test_covariates():
    """Per-study properties that cost nothing, aligned to the identifiers they describe."""
    series = pd.DataFrame({"StudyInstanceUID": ["a", "a", "b"], "SeriesInstanceUID": ["1", "2", "3"]})

    covariates = OutOfFoldOracle().covariates(["b", "a", "missing"], series)

    assert list(covariates["series_count"]) == [1.0, 2.0, 0.0], "aligned to the given order, absent means zero"


def test_lines():
    """The residual, its concentration, and the worst studies named so they can be looked at."""
    identifiers = ["quiet", "loud"]
    ours = np.zeros((2, TARGET_COUNT))
    theirs = np.zeros((2, TARGET_COUNT))
    theirs[1, 0] = 0.4
    covariates = {"series_count": np.array([1.0, 9.0])}

    lines = OutOfFoldOracleReport(identifiers, ours, theirs, covariates).lines()
    text = "\n".join(lines)

    assert "max|diff| 0.400000" in text
    assert "studies above 0.01: 1 of 2" in text
    assert "corr(residual, series_count) = +1.000" in text, "a defect that tracks a covariate says so"
    assert text.index("loud") < text.index("quiet"), "worst first, because that is what gets opened"


def test_routing(tmp_path: Path):
    """Per-study routing covariates, on a study whose slots are known by construction.

    These exist because `series_count` alone said "routing" without saying WHICH part of it, and one of
    them — contested slots — turned out to carry independent signal against a hypothesis I had already
    called refuted.
    """
    study = tmp_path / "s1"
    write_series(study / "a", slices=6)
    planes = {"a": "Sagittal"}

    covariates = OutOfFoldOracle().routing(["s1"], tmp_path, planes)

    assert sorted(covariates) == ["contested_slots", "empty_slots", "unrouted_series"]
    assert all(len(values) == 1 for values in covariates.values())
    assert covariates["empty_slots"][0] >= 1.0, "one sagittal series cannot fill six slots"
