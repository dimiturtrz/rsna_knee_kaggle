"""The mirror for the frozen-feature probe: every public method called, and asserted."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.findings import TARGETS
from kneemri.evaluation.feature_probe import FeatureProbe, ProbeCorpus, ProbeResult
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


def test_scored() -> None:
    generator = np.random.default_rng(7)
    signal = generator.normal(size=(80, 1))
    features = np.hstack([signal, generator.normal(size=(80, 3))])
    truth = (signal > 0).astype(np.int64).repeat(len(TARGETS), axis=1)
    result = FeatureProbe(penalty=1.0).scored("t", features, truth, holdout=np.arange(60, 80), training=np.arange(60))
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


def test_identifiers(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, ["b", "a"], "features_demo")
    assert corpus.identifiers("features_demo") == ["b", "a"]


def test_features(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, ["b", "a"], "features_demo")
    assert corpus.features("features_demo").shape == (2, 2, 2, 3)
