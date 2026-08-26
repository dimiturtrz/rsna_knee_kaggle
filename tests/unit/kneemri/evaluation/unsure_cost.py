"""A reader whose unsure cells are known by construction, against annotations that are known too."""

from pathlib import Path

import numpy as np
import pandas as pd

from core.data.labels.confident_labels import UNSURE
from core.findings import TARGETS
from kneemri.evaluation.unsure_cost import UNSURE_VALUE, UnsureCost, UnsureSplit

STUDIES = 10
DRAWS = 60
"""Enough draws for a percentile, few enough that the suite stays a unit test."""

BLIND = TARGETS[0]
"""The finding the reader never decided, and which the radiologist found on every study."""

SEEN = TARGETS[1]
"""The finding the reader always decided, and which the radiologist never found."""


def _train(path: Path) -> Path:
    """Annotations on every study and every finding: `BLIND` all positive, everything else negative."""
    rows = [
        {"StudyInstanceUID": f"s{index}"} | {target: (1.0 if target == BLIND else 0.0) for target in TARGETS}
        for index in range(STUDIES)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _labels(path: Path) -> Path:
    """A reader that reports `UNK` on `BLIND`, a decided verdict on `SEEN`, and no verdict elsewhere."""
    rows = [
        {"StudyInstanceUID": f"s{index}"}
        | dict.fromkeys(TARGETS, UNSURE_VALUE)
        | {f"{BLIND}__verdict": UNSURE, f"{SEEN}__verdict": "YES"}
        for index in range(STUDIES)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _measured(tmp_path: Path) -> tuple[UnsureCost, np.ndarray, np.ndarray]:
    """The cost instrument with the annotations and the unsure mask it reads them through."""
    cost = UnsureCost()
    identifiers, annotations = cost.gap.annotated(_train(tmp_path / "train.csv"))
    return cost, annotations, cost.unsure(_labels(tmp_path / "labels.csv"), identifiers)


def test_unsure(tmp_path: Path):
    """Only the finding marked `UNK` reads as unsure; a finding with no verdict column reads decided."""
    _, annotations, unsure = _measured(tmp_path)

    assert unsure.shape == annotations.shape
    assert unsure[:, TARGETS.index(BLIND)].all()
    assert not unsure[:, TARGETS.index(SEEN)].any()
    assert not unsure[:, TARGETS.index(TARGETS[2])].any(), "no verdict column is not the same as unsure"


def test_split(tmp_path: Path):
    """The blind finding's unsure cells are every study and every one of them is positive."""
    cost, annotations, unsure = _measured(tmp_path)

    split = cost.split(annotations, unsure, BLIND, TARGETS.index(BLIND))
    assert (split.unsure_cells, split.decided_cells) == (STUDIES, 0)
    assert split.unsure_positive == 1.0
    assert np.isnan(split.decided_positive), "no decided cell means no rate, not a zero"


def test_taught_error(tmp_path: Path):
    """The taught error is the distance from the default to what the radiologist actually found."""
    cost, annotations, unsure = _measured(tmp_path)

    split = cost.split(annotations, unsure, BLIND, TARGETS.index(BLIND))
    assert split.taught_error == 1.0 - UNSURE_VALUE
    assert UnsureSplit(BLIND, UNSURE_VALUE, 0.0, 1, 1).taught_error == 0.0


def test_overall(tmp_path: Path):
    """Unsure cells here are all positive and decided ones all negative, so the gap is a full unit."""
    cost, annotations, unsure = _measured(tmp_path)

    point, (floor, ceiling) = cost.overall(annotations, unsure, resamples=DRAWS)
    assert point == 1.0
    assert (floor, ceiling) == (1.0, 1.0), "every draw sees the same constructed split"


def test_report(tmp_path: Path):
    """Every finding gets a line, the blind one leads on cell count, and the verdict follows the interval."""
    cost, annotations, unsure = _measured(tmp_path)

    lines = cost.report(annotations, unsure, resamples=DRAWS)

    assert f"{STUDIES} annotated studies" in lines[0]
    assert lines[1].startswith(BLIND), "the most-unsure finding is reported first"
    assert any("pooled unsure - decided: +1.0000" in line and line.endswith("SEPARATED") for line in lines)
