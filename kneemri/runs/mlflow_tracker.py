"""The run record mirrored into an MLflow tracking store."""

import importlib
from collections.abc import Mapping
from operator import attrgetter
from pathlib import Path
from typing import cast

from kneemri.runs.mlflow_client import MlflowClient

EXPERIMENT = "kneemri"
RUN_ID_TAG = "kneemri.run_id"
GIT_TAG = "kneemri.git"
RUN_NAME_TAG = "mlflow.runName"


class MlflowTracker:
    """The record mirrored into an MLflow tracking store, one MLflow run per training run.

    Nested row fields flatten to slash-joined names — `per_label/ACL`, `lr/0` — which MLflow's own UI
    groups by prefix.
    """

    def __init__(self, client: MlflowClient, experiment: str = EXPERIMENT) -> None:
        self.client = client
        self.experiment = experiment
        self.run = ""

    @classmethod
    def opened(cls, uri: str) -> "MlflowTracker":
        """A tracker writing to the store at `uri`; raises ModuleNotFoundError when mlflow is not installed."""
        return cls(cast("MlflowClient", importlib.import_module("mlflow.tracking").MlflowClient(tracking_uri=uri)))

    def begun(self, name: str, recipe: Mapping[str, str], run: Mapping[str, object]) -> None:
        """Open an MLflow run tagged with our id and commit, and send the whole recipe as its params."""
        found = self.client.get_experiment_by_name(self.experiment)
        experiment = (
            str(attrgetter("experiment_id")(found)) if found else self.client.create_experiment(self.experiment)
        )
        tags = {RUN_NAME_TAG: name, RUN_ID_TAG: str(run.get("id", "")), GIT_TAG: str(run.get("git", ""))}
        self.run = str(attrgetter("info.run_id")(self.client.create_run(experiment, tags=tags)))
        for key, value in recipe.items():
            self.client.log_param(self.run, key, value)

    def logged(self, row: Mapping[str, object]) -> None:
        """Every numeric field of the row, flattened, at the row's epoch as the step."""
        for key, value in self.scalars(row).items():
            self.client.log_metric(self.run, key, value, step=int(cast("int", row["epoch"])))

    def attached(self, path: Path) -> None:
        """Copy one of the run's files into the store."""
        self.client.log_artifact(self.run, str(path))

    def ended(self) -> None:
        """Mark the MLflow run finished; one that never reaches this stays RUNNING, which is what a killed run is."""
        self.client.set_terminated(self.run)

    @staticmethod
    def scalars(row: Mapping[str, object], prefix: str = "") -> dict[str, float]:
        """The row's numbers under slash-joined names; strings and booleans are not metrics."""
        flat: dict[str, float] = {}
        for key, value in row.items():
            nested = dict(enumerate(value)) if isinstance(value, list) else value
            if isinstance(nested, Mapping):
                flat |= MlflowTracker.scalars({str(inner): item for inner, item in nested.items()}, f"{prefix}{key}/")
            elif isinstance(nested, int | float) and not isinstance(nested, bool):
                flat[f"{prefix}{key}"] = float(nested)
        return flat
