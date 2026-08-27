"""Two readers whose per-finding AUCs are known by construction, and a null whose answer is known too."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from core.findings import TARGETS
from kneemri.evaluation.foreign_ruler import ForeignReading, ForeignRuler, OracleNull, StudyOrderError

STUDIES = 12
DRAWS = 400
"""Enough draws for a quantile to be stable to a percent, few enough that the suite stays a unit test."""


def _truth(index: int, column: int) -> float:
    """A checkerboard, so every finding carries six positives and six negatives and none is degenerate."""
    return float((index + column) % 2)


def _annotations() -> np.ndarray:
    """The gold block every reading in this module is scored against."""
    return np.array(
        [[_truth(index, column) for column in range(len(TARGETS))] for index in range(STUDIES)],
        dtype=np.float32,
    )


def _identifiers() -> list[str]:
    """The study order the cache manifests below claim to have been written in."""
    return [f"s{index}" for index in range(STUDIES)]


def _cache(path: Path, studies: list[str], *, inverted: bool) -> Path:
    """One cached member: a probability block beside the manifest naming its checkpoint and study order."""
    path.mkdir(exist_ok=True)
    values = _annotations()
    block = (1.0 - values) if inverted else values
    name = "inverted" if inverted else "perfect"
    np.save(path / f"{name}.npy", block)
    (path / f"{name}.json").write_text(
        json.dumps(
            {
                "checkpoint": f"{name}_s2026.pt",
                "studies": hashlib.sha256("\n".join(studies).encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return path


def _checkpoint(path: Path, value: float) -> Path:
    """A foreign `.pt` publishing the same AUC on every finding, readable with `weights_only=True`."""
    torch.save({"aucs": dict.fromkeys(TARGETS, value)}, path)
    return path


def test_load(tmp_path: Path):
    """A checkpoint's published AUCs come back in `TARGETS` order; one without them comes back `None`."""
    reading = ForeignReading.load(_checkpoint(tmp_path / "foreign.pt", 0.8))

    assert reading is not None
    assert reading.name == "foreign"
    assert reading.per_finding.shape == (len(TARGETS),)
    assert abs(reading.macro - 0.8) < 1e-9

    torch.save({"model": {}}, tmp_path / "silent.pt")
    assert ForeignReading.load(tmp_path / "silent.pt") is None, "no aucs key means nothing to join on"


def test_explained():
    """The verdict reads the null's quantile, so a gain a hair over the null's MEAN is still noise."""
    assert OracleNull(0.019, 0.016, chance=0.34).explained, "a third of draws reach it, so noise stands"
    assert not OracleNull(0.019, 0.016, chance=0.001).explained


def test_foreign(tmp_path: Path):
    """Only the checkpoints that publish AUCs are returned, keyed by file stem."""
    _checkpoint(tmp_path / "published.pt", 0.9)
    torch.save({"model": {}}, tmp_path / "silent.pt")

    assert set(ForeignRuler().foreign(tmp_path)) == {"published"}


def test_ours(tmp_path: Path):
    """A perfect member scores 1.0 on every finding; a cache written over other studies raises."""
    studies = _identifiers()
    cache = _cache(tmp_path / "cache", studies, inverted=False)

    readings = ForeignRuler().ours(cache, studies, _annotations())

    assert set(readings) == {"perfect_s2026"}, "the key is the checkpoint's stem, not the block's filename"
    assert np.allclose(readings["perfect_s2026"], 1.0)

    try:
        ForeignRuler().ours(cache, [*studies[1:], "elsewhere"], _annotations())
    except StudyOrderError:
        return
    raise AssertionError("a cache scored over a different study list must raise, not print chance")


def test_against():
    """The truth read forwards scores 1.0 per finding and read backwards scores 0.0."""
    annotations = _annotations()
    ruler = ForeignRuler()

    assert np.allclose(ruler.against(annotations, annotations), 1.0)
    assert np.allclose(ruler.against(1.0 - annotations, annotations), 0.0)


def test_standard_error():
    """Hanley-McNeil is positive, and shrinks as the corpus behind the same AUC grows."""
    ruler = ForeignRuler()
    small = ruler.standard_error(0.9, 10, 10)
    large = ruler.standard_error(0.9, 100, 100)

    assert small > 0.0
    assert large < small, "ten times the studies must tighten the bracket"


def test_oracle():
    """Two equal models routed on noise gain nothing real; two genuinely split models do."""
    ruler = ForeignRuler()
    equal = np.full(len(TARGETS), 0.9)
    errors = np.full(len(TARGETS), 0.04)

    noise = ruler.oracle(equal, equal.copy(), errors, draws=DRAWS)
    assert noise.gain == 0.0
    assert noise.explained, "identical models cannot route to a prize"

    half = len(TARGETS) // 2
    ours = np.concatenate([np.full(half, 0.99), np.full(len(TARGETS) - half, 0.60)])
    theirs = ours[::-1].copy()
    split = ruler.oracle(ours, theirs, np.full(len(TARGETS), 1e-6), draws=DRAWS)

    assert split.gain > 0.15
    assert not split.explained, "a noiseless anti-correlated split is exactly what routing is for"


def test_report():
    """Every finding gets a row, the macro is the mean of the twelve, and the oracle carries its null."""
    annotations = _annotations()
    ruler = ForeignRuler()
    ours = ruler.against(annotations, annotations)
    theirs = ForeignReading("foreign", 0.8, np.full(len(TARGETS), 0.8))

    lines = ruler.report(ours, theirs, annotations, name="perfect")

    assert sum(line.startswith(target) for line in lines for target in TARGETS) == len(TARGETS)
    macro = next(line for line in lines if line.startswith("MACRO"))
    assert "1.0000" in macro and "0.8000" in macro
    assert any("oracle" in line and "chance reaches it" in line for line in lines)
