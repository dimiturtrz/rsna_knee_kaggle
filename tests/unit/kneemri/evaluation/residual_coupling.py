"""Unit tests for the residual-coupling matrix.

Every fixture is built here rather than read from the corpus: this module measures how much label
structure a model left behind, and a test that measured that on real data would be the analysis rather
than a test of it. The equivalence classes are the ones the reading can get wrong — a decided/undecided
mask, a pair no study decided, a constant column, and the ordering the matrix is reported in.
"""

from pathlib import Path

import numpy as np
import pytest

from kneemri.evaluation.residual_coupling import LabelCoupling
from kneemri.findings import TARGETS

STUDIES = 40


def table(path: Path, decided: list[list[bool]]) -> Path:
    """A label table shaped like `report_labels_v2.csv`, with `decided` driving the verdict columns."""
    header = ["StudyInstanceUID"]
    for target in TARGETS:
        header += [target, f"{target}__conf", f"{target}__verdict"]
    rows = [",".join(header)]
    for index in range(len(decided)):
        fields = [f"study{index}"]
        for column in range(len(TARGETS)):
            verdict = "YES" if decided[index][column] else "UNK"
            fields += ["0.94" if decided[index][column] else "0.28", "0.95", verdict]
        rows.append(",".join(fields))
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def alternating() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A model that beats chance but is not perfect, which is the class this reading is ever taken on."""
    steps = np.linspace(0.05, 0.95, STUDIES)
    truth = np.zeros((STUDIES, len(TARGETS)), dtype=np.float32)
    truth[STUDIES // 2 :, :] = 1.0
    truth[[1, 3], :] = 1.0
    truth[[-2, -4], :] = 0.0
    predictions = np.tile(steps.reshape(-1, 1), (1, len(TARGETS)))
    known = np.ones((STUDIES, len(TARGETS)), dtype=np.bool_)
    return predictions, truth, known


def test_decided(tmp_path: Path) -> None:
    decided = [[(index + column) % 2 == 0 for column in range(len(TARGETS))] for index in range(4)]
    mask = LabelCoupling.decided(table(tmp_path / "labels.csv", decided), ["study2", "study0"])
    assert mask.shape == (2, len(TARGETS))
    assert mask[0][0] is np.True_
    assert mask[1][1] is np.False_


def test_decided_raises_on_a_study_the_table_never_saw(tmp_path: Path) -> None:
    decided = [[True] * len(TARGETS)]
    with pytest.raises(KeyError, match="absent"):
        LabelCoupling.decided(table(tmp_path / "labels.csv", decided), ["studyX"])


def test_correlation() -> None:
    values = np.array([0.0, 1.0, 2.0, 3.0])
    assert LabelCoupling.correlation(values, values) == pytest.approx(1.0)
    assert LabelCoupling.correlation(values, -values) == pytest.approx(-1.0)
    assert np.isnan(LabelCoupling.correlation(values, np.ones(4)))


def test_pair() -> None:
    predictions, truth, known = alternating()
    known[:, 1] = False
    reading = LabelCoupling(predictions, truth, known).pair(0, 1)
    assert reading.studies == 0
    assert np.isnan(reading.truth)


def test_pairs() -> None:
    predictions, truth, known = alternating()
    readings = LabelCoupling(predictions, truth, known).pairs()
    assert len(readings) == len(TARGETS) * (len(TARGETS) - 1) // 2
    assert abs(readings[0].residual) >= abs(readings[-1].residual)
