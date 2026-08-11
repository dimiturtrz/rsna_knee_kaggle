"""Striking out a reader's unsure calls, on a table built by hand."""

import pandas as pd

from kneemri.confident_labels import ConfidentLabels
from kneemri.model import TARGETS


def _table() -> pd.DataFrame:
    """Two studies: the first asserted on every target, the second unsure on every target."""
    row_sure = {"StudyInstanceUID": "a"}
    row_unsure = {"StudyInstanceUID": "b"}
    for target in TARGETS:
        row_sure |= {target: 0.9, f"{target}__conf": 0.9, f"{target}__verdict": "YES"}
        row_unsure |= {target: 0.28, f"{target}__conf": 0.05, f"{target}__verdict": "UNK"}
    return pd.DataFrame([row_sure, row_unsure])


def test_struck():
    """An UNK target loses its VALUE — which is what makes Targets.load give it no weight."""
    struck = ConfidentLabels(_table()).struck()
    assert struck.loc[0, TARGETS[0]] == 0.9
    assert pd.isna(struck.loc[1, TARGETS[0]])


def test_struck_leaves_the_confidence_column_alone():
    """Zeroing confidence instead would leave the weight floor pulling toward a value never asserted."""
    struck = ConfidentLabels(_table()).struck()
    assert struck.loc[1, f"{TARGETS[0]}__conf"] == 0.05


def test_struck_share():
    """Half these rows are unsure on every target."""
    assert ConfidentLabels(_table()).struck_share() == dict.fromkeys(TARGETS, 0.5)
