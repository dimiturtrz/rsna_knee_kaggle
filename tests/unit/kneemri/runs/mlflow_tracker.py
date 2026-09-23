import importlib
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from kneemri.runs.mlflow_tracker import GIT_TAG, RUN_ID_TAG, RUN_NAME_TAG, MlflowTracker


class Client:
    def __init__(self, *, known: bool = False) -> None:
        self.known = known
        self.calls: list[tuple[object, ...]] = []

    def get_experiment_by_name(self, name: str) -> SimpleNamespace | None:
        return SimpleNamespace(experiment_id="7") if self.known else None

    def create_experiment(self, name: str) -> str:
        self.calls.append(("experiment", name))
        return "9"

    def create_run(self, experiment: str, tags: Mapping[str, str]) -> SimpleNamespace:
        self.calls.append(("run", experiment, tags))
        return SimpleNamespace(info=SimpleNamespace(run_id="r1"))

    def log_param(self, run: str, key: str, value: str) -> None:
        self.calls.append(("param", run, key, value))

    def log_metric(self, run: str, key: str, value: float, step: int) -> None:
        self.calls.append(("metric", run, key, value, step))

    def log_artifact(self, run: str, path: str) -> None:
        self.calls.append(("artifact", run, path))

    def set_terminated(self, run: str) -> None:
        self.calls.append(("ended", run))


def test_mlflow_tracker_begun():
    """A new experiment is created once, the run carries our id and commit as tags, the recipe goes as params."""
    client = Client()
    tracker = MlflowTracker(client)
    tracker.begun("member", {"lr": "1e-4"}, {"id": "abc", "git": "deadbeef"})

    assert tracker.run == "r1"
    assert ("experiment", "kneemri") in client.calls
    assert ("run", "9", {RUN_NAME_TAG: "member", RUN_ID_TAG: "abc", GIT_TAG: "deadbeef"}) in client.calls
    assert ("param", "r1", "lr", "1e-4") in client.calls

    known = Client(known=True)
    MlflowTracker(known).begun("member", {}, {})
    assert known.calls[0][1] == "7", "an existing experiment is reused, not re-created"


def test_mlflow_tracker_logged():
    """Every numeric field lands at the row's epoch as the step."""
    client = Client()
    tracker = MlflowTracker(client)
    tracker.run = "r1"
    tracker.logged({"epoch": 3, "auc": 0.8, "per_label": {"ACL": 0.7}})

    assert ("metric", "r1", "auc", 0.8, 3) in client.calls
    assert ("metric", "r1", "per_label/ACL", 0.7, 3) in client.calls


def test_mlflow_tracker_attached(tmp_path: Path):
    """A file the run wrote is copied into the store under the run."""
    client = Client()
    tracker = MlflowTracker(client)
    tracker.run = "r1"
    tracker.attached(tmp_path / "member.metrics.jsonl")

    assert client.calls == [("artifact", "r1", str(tmp_path / "member.metrics.jsonl"))]


def test_mlflow_tracker_ended():
    """The run is marked finished."""
    client = Client()
    tracker = MlflowTracker(client)
    tracker.run = "r1"
    tracker.ended()

    assert client.calls == [("ended", "r1")]


def test_scalars():
    """Nested maps and lists flatten to slash names; strings and booleans are not metrics."""
    flat = MlflowTracker.scalars({"auc": 0.8, "lr": [1e-5, 1e-4], "per_label": {"ACL": 1}, "name": "x", "flag": True})

    assert flat == {"auc": 0.8, "lr/0": 1e-5, "lr/1": 1e-4, "per_label/ACL": 1.0}


def test_opened(monkeypatch: pytest.MonkeyPatch):
    """The client comes from mlflow by name, pointed at the given store."""
    seen: dict[str, str] = {}

    def client(tracking_uri: str) -> Client:
        seen["uri"] = tracking_uri
        return Client()

    monkeypatch.setattr(importlib, "import_module", lambda name: SimpleNamespace(MlflowClient=client))
    tracker = MlflowTracker.opened("file:///runs")

    assert isinstance(tracker.client, Client)
    assert seen == {"uri": "file:///runs"}
