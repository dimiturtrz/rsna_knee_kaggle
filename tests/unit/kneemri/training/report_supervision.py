from pathlib import Path

import numpy as np
import pytest
import torch

from kneemri.training.report_supervision import NoReportSupervision, ReportSupervision

SEED = 2026
DIMS = 8


def _vectors(count: int = 6) -> np.ndarray:
    generator = np.random.default_rng(SEED)
    raw = generator.normal(size=(count, DIMS)).astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def test_call():
    """The term falls to zero when the prediction points at the report, and is positive otherwise."""
    vectors = _vectors()
    term = ReportSupervision(vectors, weight=1.0)
    positions = np.array([0, 1, 2])

    aligned = torch.from_numpy(vectors[positions])
    assert float(term(aligned, positions)) < 1e-5

    opposed = torch.from_numpy(-vectors[positions])
    assert float(term(opposed, positions)) > 1.0, "pointing the other way must cost, not merely differ"


def test_call_reads_direction_and_not_length():
    """A cosine term must ignore how long the predicted vector is.

    The target's length carries how far a report sits from the corpus mean - report verbosity as much
    as pathology - which an image has no way to know.
    """
    vectors = _vectors()
    term = ReportSupervision(vectors, weight=1.0)
    positions = np.array([0, 1])
    aligned = torch.from_numpy(vectors[positions])
    assert torch.isclose(term(aligned, positions), term(aligned * 7.0, positions), atol=1e-5)


def test_no_report_supervision_is_a_zero():
    """The default costs nothing and still returns a tensor the loss can add."""
    zero = NoReportSupervision()(torch.randn(3, DIMS), np.array([0, 1, 2]))
    assert float(zero) == 0.0
    assert zero.shape == ()


def test_from_file(tmp_path: Path):
    """Vectors are reordered into the CORPUS's study order, and a missing study is fatal."""
    path = tmp_path / "vectors.npz"
    np.savez(path, identifiers=np.array(["b", "a", "c"]), vectors=_vectors(3))
    stored = np.load(path, allow_pickle=True)["vectors"]

    reordered = ReportSupervision.from_file(path, ["a", "b", "c"]).vectors
    assert np.allclose(reordered[0], stored[1]), "the file's order must not be trusted over the corpus's"
    assert np.allclose(reordered[1], stored[0])

    with pytest.raises(KeyError):
        ReportSupervision.from_file(path, ["a", "missing"])
