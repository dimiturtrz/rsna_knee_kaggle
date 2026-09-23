"""What an MLflow tracker needs of MlflowClient, so the package can stay an optional import."""

from collections.abc import Mapping
from typing import Protocol


class MlflowClient(Protocol):
    """The slice of `mlflow.tracking.MlflowClient` a tracker calls, named so the package stays optional."""

    def get_experiment_by_name(self, name: str) -> object | None:
        """The experiment called `name`, carrying `experiment_id`, or None."""
        ...

    def create_experiment(self, name: str) -> str:
        """A new experiment's id."""
        ...

    def create_run(self, experiment: str, tags: Mapping[str, str]) -> object:
        """A new run, carrying `info.run_id`."""
        ...

    def log_param(self, run: str, key: str, value: str) -> None:
        """One hyper-parameter."""
        ...

    def log_metric(self, run: str, key: str, value: float, step: int) -> None:
        """One metric value at one step."""
        ...

    def log_artifact(self, run: str, path: str) -> None:
        """One file copied into the store."""
        ...

    def set_terminated(self, run: str) -> None:
        """The run finished."""
        ...
