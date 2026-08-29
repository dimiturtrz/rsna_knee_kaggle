"""The gold rows themselves, on a train.csv written by hand so which studies survive is known."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.findings import TARGETS
from core.paths import ENV_OVERRIDE, DataRoot
from kneemri.evaluation.gold_rows import GoldRows, main

STUDIES = 8


def _train_table(path: Path, *, annotate: int) -> Path:
    """A train.csv where the first `annotate` studies carry every finding and the rest carry none."""
    rows = []
    for index in range(STUDIES):
        row: dict[str, object] = {"StudyInstanceUID": f"s{index}"}
        for column, target in enumerate(TARGETS):
            row[target] = float((index + column) % 2) if index < annotate else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_annotated(tmp_path: Path):
    """Only the studies annotated on EVERY finding, since a partial row means nothing coherent."""
    identifiers, truth = GoldRows().annotated(_train_table(tmp_path / "train.csv", annotate=3))

    assert identifiers == ["s0", "s1", "s2"], "the unannotated studies are dropped whole"
    assert truth.shape == (3, len(TARGETS))
    assert set(np.unique(truth)) <= {0.0, 1.0}


def test_reported(tmp_path: Path):
    """The report table in the CALLER's order, with a missing study defaulted rather than dropped."""
    path = tmp_path / "labels.csv"
    rows = [{"StudyInstanceUID": f"s{index}"} | dict.fromkeys(TARGETS, 0.94 - 0.1 * index) for index in (0, 1)]
    pd.DataFrame(rows).to_csv(path, index=False)

    values = GoldRows().reported(path, ["s1", "s0", "absent"])

    assert values.shape == (3, len(TARGETS))
    assert values[0, 0] < values[1, 0], "reindexed to the order asked for, not the order on disk"
    assert values[2, 0] == 0.5, "an unlabelled study sits at the default rather than at zero"


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The identifiers land one per line, which is the shape `build_cache --only-studies` reads."""
    root = DataRoot(tmp_path)
    root.raw().mkdir(parents=True)
    _train_table(root.raw() / "train.csv", annotate=2)
    listing = tmp_path / "gold.txt"
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    monkeypatch.setattr("sys.argv", ["gold_rows", "--out", str(listing)])

    main()

    assert listing.read_text(encoding="utf-8").split() == ["s0", "s1"]
