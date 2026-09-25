"""A raptor-first entry: theirs scored first into scratch, ours on what is left, the two blended by seat."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.findings import STUDY, TARGETS
from kneemri.inference.budget import KERNEL_SECONDS
from kneemri.inference.ensemble import Serving
from kneemri.inference.vote import LogitAverage, RankAverage
from kneemri.kaggle import seated_entry
from kneemri.kaggle.raptor_seat import RaptorSeat
from kneemri.kaggle.seated_entry import SUBMISSION, SeatedEntry

STUDIES = ["a", "b", "c", "d"]


def _table(values: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(np.repeat(values[:, None], len(TARGETS), axis=1), columns=list(TARGETS))
    frame.insert(0, STUDY, STUDIES)
    return frame


OURS = _table(np.linspace(0.1, 0.9, len(STUDIES)))
THEIRS = _table(np.linspace(0.9, 0.2, len(STUDIES)))
OPENED: list[str] = []
SCRATCHES: list[Path] = []
BUILT: list["_Run"] = []


class _Mounts:
    def one(self, name: str, _what: str) -> Path:
        return Path(name)


class _Cards:
    def device(self) -> str:
        return "cpu"


class _Member:
    @staticmethod
    def loaded(weights: Path, device: str) -> tuple[Path, str]:
        return weights, device


class _Entry:
    @staticmethod
    def weights(filename: str, _mounts: object) -> Path:
        OPENED.append(filename)
        return Path(filename)

    def __init__(self, _sample: Path, working: Path) -> None:
        SCRATCHES.append(working)

    def run(self, member: tuple[Path, str], device: str) -> pd.DataFrame:
        assert member == (Path(OPENED[-1]), device)
        return THEIRS


@dataclass
class _Run:
    names: tuple[str, ...]
    overlap: bool
    serving: Serving

    def run(self) -> None:
        BUILT.append(self)
        OURS.to_csv(seated_entry.WORKING / SUBMISSION, index=False)


@pytest.fixture
def entry_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for record in (OPENED, SCRATCHES, BUILT):
        record.clear()
    for name, fake in {
        "WORKING": tmp_path,
        "Mounts": _Mounts,
        "VisibleCards": _Cards,
        "RaptorMember": _Member,
        "RaptorEntry": _Entry,
        "KernelRun": _Run,
    }.items():
        monkeypatch.setattr(seated_entry, name, fake)
    return tmp_path


def test_run(entry_module: Path) -> None:
    """Ours is served on the session's remainder and overwritten by the seat blend, theirs from scratch."""
    vote = RankAverage()
    SeatedEntry(names=("x", "y"), vote=vote, weight=1.0 / 3.0, weights="raptor_ft_effv2l480.pt").run()

    (run,) = BUILT
    assert run.names == ("x", "y")
    assert run.overlap is False
    assert run.serving.vote is vote
    assert run.serving.budget is not None
    assert 0.0 <= run.serving.budget.seconds <= KERNEL_SECONDS
    assert OPENED == ["raptor_ft_effv2l480.pt"]
    assert [entry_module / seated_entry.SCRATCH] == SCRATCHES
    assert (entry_module / seated_entry.SCRATCH).is_dir()

    written = pd.read_csv(entry_module / SUBMISSION)
    expected = RaptorSeat(weight=1.0 / 3.0).blended(OURS, THEIRS)
    pd.testing.assert_frame_equal(written, expected, check_dtype=False)


def test_run_defaults(entry_module: Path) -> None:
    """With nothing but names: the default checkpoint, one seat's share, a logit average, no overlap."""
    SeatedEntry(names=("x",)).run()

    assert OPENED == [seated_entry.DEFAULT_WEIGHTS]
    assert isinstance(BUILT[0].serving.vote, LogitAverage)
    written = pd.read_csv(entry_module / SUBMISSION)
    pd.testing.assert_frame_equal(written, RaptorSeat().blended(OURS, THEIRS), check_dtype=False)
