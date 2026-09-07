"""His out-of-fold arrays become one of our records, joined by identifier and judged by OUR table."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.findings import TARGETS
from kneemri.evaluation.foreign_holdout import IDENTIFIERS, PREDICTIONS, UNKNOWN_EPOCH, ForeignHoldout
from kneemri.runs.holdout_record import FOREIGN, HoldoutRecord

STUDIES = 6


def _oof(path: Path, identifiers: list[str]) -> Path:
    """His file, spelled his way: `pred` and `ids`, plus a `truth` that is HIS table and must be ignored."""
    rng = np.random.default_rng(2026)
    np.savez(
        path,
        pred=rng.random((len(identifiers), len(TARGETS))),
        truth=np.zeros((len(identifiers), len(TARGETS)), dtype=np.float32),
        ids=np.array(identifiers),
        fold=0,
        nfolds=5,
    )
    return path


def _labels(path: Path, identifiers: list[str]) -> Path:
    """Our table, in a DIFFERENT order from his file, which is the whole point of joining by name."""
    frame = pd.DataFrame({"StudyInstanceUID": list(reversed(identifiers))})
    for offset, target in enumerate(TARGETS):
        frame[target] = [0.9 if (index + offset) % 2 else 0.1 for index in range(len(identifiers))]
    frame.to_csv(path, index=False)
    return path


def test_read(tmp_path: Path):
    """His two keys come back as ours; a file missing either is refused rather than joined by position."""
    identifiers = [f"s{index}" for index in range(STUDIES)]
    held = ForeignHoldout(source=_oof(tmp_path / "oof.npz", identifiers), labels=tmp_path / "unused.csv")

    names, predictions = held.read()

    assert names == identifiers, "identifiers come back as strings, in his order"
    assert predictions.shape == (STUDIES, len(TARGETS))
    assert predictions.dtype == np.float64

    headless = tmp_path / "headless.npz"
    np.savez(headless, pred=np.zeros((2, 2)))
    assert PREDICTIONS in np.load(headless), "it carries his predictions and nothing that names the rows"
    with pytest.raises(ValueError, match=IDENTIFIERS):
        ForeignHoldout(source=headless, labels=tmp_path / "unused.csv").read()


def test_written(tmp_path: Path):
    """The record carries OUR truth for HIS studies, by name, marked foreign and with no invented epoch."""
    identifiers = [f"s{index}" for index in range(STUDIES)]
    held = ForeignHoldout(
        source=_oof(tmp_path / "oof.npz", identifiers),
        labels=_labels(tmp_path / "ours.csv", identifiers),
    )
    checkpoint = tmp_path / "repl2x2_a.pt"

    held.written(checkpoint)
    reading = HoldoutRecord(checkpoint).read()

    assert reading is not None
    assert reading.studies == identifiers
    assert reading.source == FOREIGN, "never mistakeable for arrays this repo's trainer produced"
    assert reading.epoch == UNKNOWN_EPOCH, "his OOF file records the fold, not the epoch — so none is invented"
    assert not np.allclose(reading.truth, 0.0), "his own truth array is NOT what the record is scored against"
    # Our table was written in reverse order, so a positional join would give every study its mirror's
    # row — full-length, plausible, and wrong on every study but the middle two.
    ours = pd.read_csv(tmp_path / "ours.csv").set_index("StudyInstanceUID")[list(TARGETS)]
    assert np.allclose(reading.truth[0], ours.loc["s0"].to_numpy(dtype=np.float32))
