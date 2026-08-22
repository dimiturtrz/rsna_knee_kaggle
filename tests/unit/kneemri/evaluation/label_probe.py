"""Unit tests for the held-out label probe — the verdict half of the residual-coupling gate.

Every fixture is built here rather than read from the corpus: the probe's whole job is to say whether a
label interaction generalises, and a test that measured that on real data would be the analysis rather
than a test of it. The equivalence classes are the ones the probe can get wrong — a feature that can only
be reproduced (the null control), a label it may genuinely refine, an undecided mask that must shrink the
scoring half, and a record that kept no predictions at all.
"""

from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.label_probe import HeldOutProbe

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
    """A model that beats chance but is not perfect, which is the class the probe is ever run on.

    Positives sit at the HIGH predictions, because a single-feature ridge learns the sign from the data
    and would flip the ranking of a below-chance column — see `HeldOutProbe.alone`. Four studies are
    inverted so the raw AUC is short of 1.0 and a refinement has somewhere to move.
    """
    steps = np.linspace(0.05, 0.95, STUDIES)
    truth = np.zeros((STUDIES, len(TARGETS)), dtype=np.float32)
    truth[STUDIES // 2 :, :] = 1.0
    truth[[1, 3], :] = 1.0
    truth[[-2, -4], :] = 0.0
    predictions = np.tile(steps.reshape(-1, 1), (1, len(TARGETS)))
    known = np.ones((STUDIES, len(TARGETS)), dtype=np.bool_)
    return predictions, truth, known


def test_fit() -> None:
    features = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    truth = np.array([0, 0, 1, 1])
    weights = HeldOutProbe(*alternating()).fit(features, truth)
    assert weights[0] > 0.0
    assert weights[1] == pytest.approx(0.0, abs=1e-12)


def test_alone() -> None:
    """The null control must be EXACT: one monotone feature cannot reorder its own ranking."""
    probe = HeldOutProbe(*alternating())
    raw, refined, studies = probe.alone(0)
    assert studies == STUDIES // 2
    assert refined == pytest.approx(raw, abs=1e-12)


def test_label() -> None:
    predictions, truth, known = alternating()
    raw, refined, studies = HeldOutProbe(predictions, truth, known).label(0)
    assert studies == STUDIES // 2
    assert 0.0 <= refined <= 1.0
    assert 0.5 < raw < 1.0


def test_label_scores_only_the_decided_half() -> None:
    """Undecided pairs are dropped from the SCORING half, which is the odd positions — see the split."""
    predictions, truth, known = alternating()
    known[1::4, 0] = False
    assert HeldOutProbe(predictions, truth, known).label(0)[2] < STUDIES // 2


def test_report(tmp_path: Path) -> None:
    predictions, truth, _ = alternating()
    studies = [f"study{index}" for index in range(STUDIES)]
    record = tmp_path / "arm.holdout.npz"
    np.savez(record, studies=np.array(studies), predictions=predictions, truth=truth, epoch=3, source="trained")
    decided = [[True] * len(TARGETS) for _ in range(STUDIES)]
    reading = HeldOutProbe.report(record, table(tmp_path / "labels.csv", decided))
    assert reading["epoch"] == 3
    assert reading["decided_pairs"] == STUDIES * len(TARGETS)
    assert reading["null_control_max_abs"] == pytest.approx(0.0, abs=1e-9)
    assert reading["labels_scored"] == len(TARGETS)


def test_report_raises_when_the_run_kept_no_predictions(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no holdout predictions"):
        HeldOutProbe.report(tmp_path / "absent.holdout.npz", tmp_path / "labels.csv")
