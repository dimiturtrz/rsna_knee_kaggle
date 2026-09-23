import importlib
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from kneemri.runs.mlflow_tracker import MlflowTracker
from kneemri.runs.tracker import MLFLOW_ENV, Fanout


class Sink:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def begun(self, name: str, recipe: Mapping[str, str], run: Mapping[str, object]) -> None:
        self.calls.append(f"begun {name}")

    def logged(self, row: Mapping[str, object]) -> None:
        self.calls.append(f"logged {row['epoch']}")

    def attached(self, path: Path) -> None:
        self.calls.append(f"attached {path.name}")

    def ended(self) -> None:
        self.calls.append("ended")


def test_configured(monkeypatch: pytest.MonkeyPatch):
    """No store named, no mirrors; a store named, one MLflow mirror."""
    assert Fanout.configured({}).sinks == ()

    monkeypatch.setattr(
        importlib, "import_module", lambda name: SimpleNamespace(MlflowClient=lambda tracking_uri: None)
    )
    (sink,) = Fanout.configured({MLFLOW_ENV: "file:///runs"}).sinks
    assert isinstance(sink, MlflowTracker)


def test_fanout_begun():
    """Every sink begins."""
    first, second = Sink(), Sink()
    Fanout((first, second)).begun("member", {}, {})
    assert first.calls == second.calls == ["begun member"]


def test_fanout_logged():
    """Every sink sees the row."""
    first, second = Sink(), Sink()
    Fanout((first, second)).logged({"epoch": 2})
    assert first.calls == second.calls == ["logged 2"]


def test_fanout_attached(tmp_path: Path):
    """Every sink keeps the file."""
    first, second = Sink(), Sink()
    Fanout((first, second)).attached(tmp_path / "x.npz")
    assert first.calls == second.calls == ["attached x.npz"]


def test_fanout_ended():
    """Every sink ends."""
    first, second = Sink(), Sink()
    Fanout((first, second)).ended()
    assert first.calls == second.calls == ["ended"]
