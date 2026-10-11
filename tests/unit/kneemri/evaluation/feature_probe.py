"""The mirror for the frozen-feature probe: every public method called, and asserted."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data.corpus.folds import ReportFolds
from core.findings import TARGETS
from core.paths import TRAIN_TABLE
from kneemri.evaluation.feature_probe import (
    HOLDOUT_FOLD,
    SELECTION,
    TRUNKS,
    FeatureProbe,
    ProbeCorpus,
    ProbeResult,
    ProbeSplit,
)
from kneemri.evaluation.scores import LabelScores
from kneemri.evaluation.slice_readout import READOUTS, SliceMax, SliceMean


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


SPLIT_STUDIES = 40


def _split_corpus(tmp_path: Path, stem: str) -> tuple[ProbeCorpus, list[str]]:
    identifiers = [f"s{index}" for index in range(SPLIT_STUDIES)]
    corpus = _corpus(tmp_path, identifiers, stem)
    corpus.raw.mkdir()
    reports = [f"report {index}" for index in range(SPLIT_STUDIES)]
    pd.DataFrame({"StudyInstanceUID": identifiers, "Report": reports}).to_csv(corpus.raw / TRAIN_TABLE, index=False)
    features = np.random.default_rng(7).normal(size=(SPLIT_STUDIES, 2, 2, 3)).astype(np.float16)
    np.save(corpus.processed / f"{stem}.npy", features)
    return corpus, identifiers


def _manual_split(corpus: ProbeCorpus, stem: str, identifiers: list[str], readout: str) -> ProbeSplit:
    folds = ReportFolds()
    training, holdout = folds.holdout(folds.assign(str(corpus.raw / TRAIN_TABLE), identifiers), HOLDOUT_FOLD)
    features = FeatureProbe.pooled(corpus.features(stem), READOUTS[readout])
    return ProbeSplit(features, corpus.truth(identifiers), training, holdout)


def test_split(tmp_path: Path) -> None:
    """The split equals the fold-0 block the probe mains built by hand, array for array."""
    name = sorted(TRUNKS)[0]
    corpus, identifiers = _split_corpus(tmp_path, TRUNKS[name])
    expected = _manual_split(corpus, TRUNKS[name], identifiers, SliceMax.name)
    split = corpus.split(name, SliceMax.name)
    assert len(expected.training) > 0
    assert len(expected.holdout) > 0
    for field in ("features", "truth", "training", "holdout"):
        assert np.array_equal(getattr(split, field), getattr(expected, field))


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


ORACLE_STUDIES = 120
ONE_CLASS_COLUMN = 3


def _oracle_split() -> ProbeSplit:
    """Random binary truth with one finding that holds a single class on the scored positions."""
    generator = np.random.default_rng(5)
    truth = generator.integers(0, 2, size=(ORACLE_STUDIES, len(TARGETS)))
    truth[:, ONE_CLASS_COLUMN] = 0
    features = truth[:, :4] * 0.5 + generator.normal(size=(ORACLE_STUDIES, 4))
    positions = np.arange(ORACLE_STUDIES)
    return ProbeSplit(features, truth, positions[: ORACLE_STUDIES // 2], positions[ORACLE_STUDIES // 2 :])


def _rank_areas(predictions: np.ndarray, held: np.ndarray) -> dict[str, float]:
    return {
        target: LabelScores.area_under_curve(predictions[:, column], held[:, column])
        for column, target in enumerate(TARGETS)
    }


def test_scored_matches_rank_areas_with_a_one_class_column() -> None:
    split = _oracle_split()
    probe = FeatureProbe(penalty=1.0)
    expected = _rank_areas(probe.holdout_record(split), split.truth[split.holdout])
    per_label = probe.scored("t", split).per_label
    np.testing.assert_equal(per_label, expected)
    assert np.isnan(per_label[TARGETS[ONE_CLASS_COLUMN]])


def test_inner_area_is_the_mean_over_defined_labels() -> None:
    split = _oracle_split()
    probe = FeatureProbe(penalty=1.0)
    areas = _rank_areas(probe.holdout_record(split), split.truth[split.holdout])
    defined = [area for area in areas.values() if not np.isnan(area)]
    assert len(defined) == len(TARGETS) - 1
    assert probe.inner_area(split) == float(np.mean(defined))


def test_inner_area_reads_zero_when_no_label_is_defined() -> None:
    split = _oracle_split()
    flat = ProbeSplit(split.features, np.zeros_like(split.truth), split.training, split.holdout)
    assert FeatureProbe(penalty=1.0).inner_area(flat) == 0.0
