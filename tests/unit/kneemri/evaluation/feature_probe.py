"""The mirror for the frozen-feature probe: every public method called, and asserted."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.findings import TARGETS
from kneemri.evaluation.feature_probe import SELECTION, FeatureProbe, ProbeCorpus, ProbeResult, ProbeSplit
from kneemri.evaluation.slice_readout import SliceMax, SliceMean


def _result() -> ProbeResult:
    return ProbeResult(trunk="t", dimensions=4, studies=9, per_label=dict.fromkeys(TARGETS, 0.75))


def test_mean() -> None:
    assert _result().mean == 0.75


def test_line() -> None:
    line = _result().line()
    assert "t" in line
    assert "0.7500" in line


def _stack() -> np.ndarray:
    """One study whose channel varies along the stack, against one that is flat there."""
    array = np.ones((2, 2, 4, 3), dtype=np.float16)
    array[0, :, 3, :] = 5.0
    return array


def test_pooled() -> None:
    array = np.ones((3, 2, 5, 4), dtype=np.float16)
    array[0] = 3.0
    pooled = FeatureProbe.pooled(array)
    assert pooled.shape == (3, 4)
    assert pooled[0, 0] == 3.0
    assert pooled[1, 0] == 1.0


def test_pooled_takes_the_readout_it_is_given() -> None:
    """The default must stay the mean, or every trunk number on the ledger changes silently."""
    array = _stack()
    assert FeatureProbe.pooled(array)[0, 0] == FeatureProbe.pooled(array, SliceMean())[0, 0]
    assert FeatureProbe.pooled(array, SliceMax())[0, 0] == 5.0


def test_weights() -> None:
    generator = np.random.default_rng(2026)
    features = generator.normal(size=(60, 4))
    truth = (features[:, :1] > 0).astype(np.int64).repeat(len(TARGETS), axis=1)
    weights = FeatureProbe(penalty=1.0).weights(features, truth)
    assert weights.shape == (5, len(TARGETS))
    assert weights[0, 0] > abs(weights[1, 0])


def test_predicted() -> None:
    features = np.zeros((2, 3))
    weights = np.arange(4 * len(TARGETS), dtype=np.float64).reshape(4, len(TARGETS))
    predictions = FeatureProbe.predicted(features, weights)
    assert predictions.shape == (2, len(TARGETS))
    assert np.allclose(predictions[0], weights[-1])


def test_holdout_record() -> None:
    generator = np.random.default_rng(11)
    features = generator.normal(size=(40, 3))
    truth = (features[:, :1] > 0).astype(np.int64).repeat(len(TARGETS), axis=1)
    holdout = np.arange(10, dtype=np.int64)
    training = np.arange(10, 40, dtype=np.int64)
    record = FeatureProbe(penalty=1.0).holdout_record(ProbeSplit(features, truth, training, holdout))
    assert record.shape == (len(holdout), len(TARGETS))
    assert np.isfinite(record).all()


def test_fitted() -> None:
    """Weights come from the training rows alone: corrupting the holdout rows leaves them unchanged."""
    generator = np.random.default_rng(5)
    features = generator.normal(size=(40, 3))
    truth = (features[:, :1] > 0).astype(np.int64).repeat(len(TARGETS), axis=1)
    holdout, training = np.arange(10, dtype=np.int64), np.arange(10, 40, dtype=np.int64)
    probe = FeatureProbe(penalty=1.0)
    clean = probe.fitted(ProbeSplit(features, truth, training, holdout))
    corrupted = features.copy()
    corrupted[holdout] = 1e6
    assert clean.shape == (features.shape[1] + 1, len(TARGETS))
    assert np.allclose(clean, probe.fitted(ProbeSplit(corrupted, truth, training, holdout)))


def test_scored() -> None:
    generator = np.random.default_rng(7)
    signal = generator.normal(size=(80, 1))
    features = np.hstack([signal, generator.normal(size=(80, 3))])
    truth = (signal > 0).astype(np.int64).repeat(len(TARGETS), axis=1)
    result = FeatureProbe(penalty=1.0).scored(
        "t", ProbeSplit(features, truth, training=np.arange(60), holdout=np.arange(60, 80))
    )
    assert result.studies == 20
    assert result.dimensions == 4
    assert result.mean > 0.9


def _corpus(tmp_path: Path, identifiers: list[str], stem: str) -> ProbeCorpus:
    processed = tmp_path / "processed"
    (processed / "labels").mkdir(parents=True)
    frame = pd.DataFrame({"StudyInstanceUID": identifiers})
    for column, target in enumerate(TARGETS):
        frame[target] = float(column % 2)
    frame.to_csv(processed / "labels" / "report_labels_v2.csv", index=False)
    index = processed / f"features_index_{stem.removeprefix('features_')}.json"
    index.write_text(json.dumps(identifiers), encoding="utf-8")
    np.save(processed / f"{stem}.npy", np.ones((len(identifiers), 2, 2, 3), dtype=np.float16))
    return ProbeCorpus(processed=processed, raw=tmp_path / "raw")


def test_truth(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, ["b", "a"], "features_demo")
    truth = corpus.truth(["a", "b"])
    assert truth.shape == (2, len(TARGETS))
    assert truth[0, 0] == 0
    assert truth[0, 1] == 1


def test_graded(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    (processed / "labels").mkdir(parents=True)
    (tmp_path / "raw").mkdir()
    frame = pd.DataFrame({"StudyInstanceUID": ["a", "b"]})
    for target in TARGETS:
        frame[target] = [0.68, 0.28]
    frame.to_csv(processed / "labels" / "report_labels_v2.csv", index=False)
    annotated = pd.DataFrame({"StudyInstanceUID": ["a"], TARGETS[0]: [1.0]})
    annotated.to_csv(tmp_path / "raw" / "train.csv", index=False)

    graded = ProbeCorpus(processed=processed, raw=tmp_path / "raw").graded(["a", "b"])

    assert graded[0, 1] == pytest.approx(0.68)
    assert graded[1, 1] == pytest.approx(0.28)
    assert graded[0, 0] == pytest.approx(1.0)


def test_identifiers(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, ["b", "a"], "features_demo")
    assert corpus.identifiers("features_demo") == ["b", "a"]


def test_features(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, ["b", "a"], "features_demo")
    assert corpus.features("features_demo").shape == (2, 2, 2, 3)


def _probe_pool(studies: int, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    truth = (rng.random((studies, len(TARGETS))) > 0.6).astype(np.int64)
    features = truth[:, :1] * 0.8 + rng.normal(0.0, 0.4, (studies, dim))
    return features, truth, np.arange(studies)


def test_chosen() -> None:
    """The penalty comes from the grid, and the caller's holdout never enters the choice."""
    features, truth, training = _probe_pool(studies=240, dim=6)
    assert FeatureProbe.chosen(ProbeSplit(features, truth, training, np.arange(0))) in set(SELECTION)


def test_inner_area() -> None:
    """A separable pool reads above chance on the inner scoring positions."""
    features, truth, training = _probe_pool(studies=240, dim=6)
    assert FeatureProbe(1.0).inner_area(ProbeSplit(features, truth, training[:120], training[120:])) > 0.5
