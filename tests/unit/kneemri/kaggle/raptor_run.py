"""The raptor entry's table assembly and its one-study fallback, on tables written here.

The scoring itself is proven elsewhere — `raptor_model`'s oracle compares probabilities against theirs
element for element — so what is left to get wrong here is the part no oracle covers: the column order
of the written table, and what happens to a study whose pixels will not read.
"""

import logging
from pathlib import Path
from typing import cast, override

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor, nn

from core.findings import STUDY, TARGETS
from core.mounts import Mounts
from kneemri.evaluation.raptor_feed import RaptorFeed
from kneemri.evaluation.raptor_model import Architecture, RaptorClassifier, RaptorMember
from kneemri.kaggle.raptor_run import DEFAULT_WEIGHTS, FALLBACK_PROBABILITY, RaptorEntry

RESOLUTION = 16


class Refusing(nn.Module):
    """A model that always raises, so the fallback is driven by the tests rather than by luck."""

    @override
    def forward(self, windows: Tensor) -> Tensor:
        """Refuse, the way an arm served on the wrong resolution or a broken study would."""
        message = f"this model refuses every study, including one of {tuple(windows.shape)}"
        raise RuntimeError(message)


class Constant(nn.Module):
    """A model that scores every study the same, so a reading can be told apart from a fallback."""

    @override
    def forward(self, windows: Tensor) -> Tensor:
        """One logit per finding, well away from the 0.5 an unscored study contributes."""
        return torch.full((windows.shape[0], len(TARGETS)), 4.0)


def member() -> RaptorMember:
    """A member that cannot score anything — every test here drives the FAILING path on purpose."""
    return RaptorMember(cast("RaptorClassifier", Refusing()), Architecture("resnet18", RESOLUTION))


def competition(root: Path, studies: list[str]) -> Path:
    """A split with tables and no pixels at all, which is the fallback path written down."""
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({STUDY: studies}).to_csv(root / "test.csv", index=False)
    pd.DataFrame(
        {
            STUDY: studies,
            "SeriesInstanceUID": [f"{study}-series" for study in studies],
            "Anatomical_Plane": ["Sagittal"] * len(studies),
            "Fluid_Sensitive": ["1"] * len(studies),
        }
    ).to_csv(root / "test_series.csv", index=False)
    (root / "test_series").mkdir(exist_ok=True)
    return root


def test_weights(tmp_path: Path) -> None:
    """The checkpoint is found by name under the mounts, not assembled from a guessed path."""
    (tmp_path / "raptor" / "arms").mkdir(parents=True)
    (tmp_path / "raptor" / "arms" / DEFAULT_WEIGHTS).write_bytes(b"not really a checkpoint")

    found = RaptorEntry.weights(mounts=Mounts(root=tmp_path))

    assert found == tmp_path / "raptor" / "arms" / DEFAULT_WEIGHTS


def test_tables(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Two members, ONE decode of the split, and a frame each in the submission's own column order.

    One study, because this fixture has no pixels: every decode falls back, and a second study would
    make the fallback CONSTANT down each column, which `table` refuses on purpose. The equivalence class
    here is the fan-out — one split read, one frame per member — not the fallback, which `test_readings`
    already covers over two studies.
    """
    studies = ["study-a"]
    entry = RaptorEntry(competition(tmp_path / "comp", studies), working=tmp_path)
    scoring = RaptorMember(cast("RaptorClassifier", Constant()), Architecture("resnet18", RESOLUTION))

    with caplog.at_level(logging.ERROR):
        frames = entry.tables({"refusing": member(), "scoring": scoring}, "cpu")

    assert list(frames) == ["refusing", "scoring"]
    for frame in frames.values():
        assert list(frame.columns) == [STUDY, *TARGETS]
        assert frame[STUDY].tolist() == studies
    assert len(caplog.records) == len(studies), "only the refusing member may be charged for a failure"


def test_readings(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Each member gets its OWN array out of one decode, and one member's failure stays that member's.

    The equivalence class this covers is the one a fleet pass introduces and a single pass cannot have:
    a checkpoint that will not run beside twelve that will. Charging its failure to the shared decode
    would hand the other twelve a corpus of 0.5s, which trains and serves and scores around chance.
    """
    studies = ["study-a", "study-b"]
    entry = RaptorEntry(competition(tmp_path / "comp", studies), working=tmp_path)
    scoring = RaptorMember(cast("RaptorClassifier", Constant()), Architecture("resnet18", RESOLUTION))

    with caplog.at_level(logging.ERROR):
        rows = entry.readings(studies, {}, {"refusing": member(), "scoring": scoring}, "cpu")

    assert list(rows) == ["refusing", "scoring"]
    assert np.array_equal(rows["refusing"], np.full((len(studies), len(TARGETS)), FALLBACK_PROBABILITY))
    assert not np.any(rows["scoring"] == FALLBACK_PROBABILITY)
    assert len(caplog.records) == len(studies), "only the refusing member may be charged for a failure"


def test_table(tmp_path: Path) -> None:
    """The columns are `TARGETS` in order, behind the study id — the submission's own layout."""
    studies = ["study-a", "study-b"]
    entry = RaptorEntry(competition(tmp_path / "comp", studies), working=tmp_path)
    probabilities = np.linspace(0.0, 1.0, len(studies) * len(TARGETS)).reshape(len(studies), len(TARGETS))

    frame = entry.table(studies, probabilities)

    assert list(frame.columns) == [STUDY, *TARGETS]
    assert frame[STUDY].tolist() == studies


def test_table_refuses_a_constant_table(tmp_path: Path) -> None:
    """All twelve flat is the sample submission, which scored chance once and must never be written."""
    studies = ["study-a", "study-b"]
    entry = RaptorEntry(competition(tmp_path / "comp", studies), working=tmp_path)

    with pytest.raises(ValueError, match="constant"):
        entry.table(studies, np.full((len(studies), len(TARGETS)), FALLBACK_PROBABILITY))


def test_run(tmp_path: Path) -> None:
    """One study end to end: tables read, row written, and the file lands where a kernel looks."""
    entry = RaptorEntry(competition(tmp_path / "comp", ["study-a"]), working=tmp_path)

    frame = entry.run(member(), "cpu")

    written = pd.read_csv(tmp_path / "submission.csv")
    assert len(frame) == 1
    assert list(written.columns) == [STUDY, *TARGETS]
    assert written[STUDY].tolist() == ["study-a"]


class Numbered:
    """A feed whose every study decodes to its own number, except the one named broken."""

    def __init__(self, broken: str) -> None:
        self.broken = broken

    def study(self, study: str, records: object) -> tuple[np.ndarray, np.ndarray]:
        """A one-voxel volume holding the study's number, so order is visible in the output."""
        del records
        if study == self.broken:
            message = f"{study} is unreadable"
            raise OSError(message)
        return np.full((1, 1, 1), int(study), np.uint8), np.ones(1, np.uint8)


def test_decoded(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Decodes come back in INPUT order past the read-ahead window, a broken study as None in its place.

    Classes: more studies than the window (2 x one decoder), so the bounded deque must drain while still
    submitting; and one unreadable study mid-stream, which must not shift any later study's row.
    """
    studies = [str(number) for number in range(7)]
    entry = RaptorEntry(competition(tmp_path / "comp", studies), working=tmp_path, decoders=1)

    with caplog.at_level(logging.ERROR):
        decoded = list(entry.decoded(cast("RaptorFeed", Numbered("3")), studies, {}))

    assert [None if item is None else int(item[0][0, 0, 0]) for item in decoded] == [0, 1, 2, None, 4, 5, 6]
    assert len(caplog.records) == 1


def test_one() -> None:
    """A readable study is its decode; an unreadable one is None, never a raise."""
    feed = cast("RaptorFeed", Numbered("9"))

    readable = RaptorEntry.one(feed, "5", {})

    assert readable is not None
    assert int(readable[0][0, 0, 0]) == 5
    assert RaptorEntry.one(feed, "9", {}) is None
