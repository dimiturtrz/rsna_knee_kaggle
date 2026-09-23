"""Where a run's record goes besides its own JSONL: one interface, any number of mirrors.

`RunMetrics` writes the JSONL beside the checkpoint and stays the record every check reads; a sink here
only MIRRORS what it wrote, in the order it wrote it. The interface exists because its absence is what
went wrong in the project this was studied from (D:/work/ai): four sinks glued together with no common
contract, and hyper-parameters built for MLflow that its `register` never sent — a mirror nobody could
see drifting from the record it was supposed to mirror.

MLflow is loaded BY NAME and only when `KNEEMRI_MLFLOW` names a tracking URI, so a run that does not ask
for it — and the Kaggle kernel, which never imports this — cannot fail on its absence. The run's identity
stays ours: MLflow's own run id is its storage key, and our `run_id` rides along as a tag.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kneemri.runs.mlflow_tracker import MlflowTracker

MLFLOW_ENV = "KNEEMRI_MLFLOW"


class Tracker(Protocol):
    """One mirror of a run's record."""

    def begun(self, name: str, recipe: Mapping[str, str], run: Mapping[str, object]) -> None:
        """A run started under `name`, with this recipe and this identity."""
        ...

    def logged(self, row: Mapping[str, object]) -> None:
        """One epoch's row, exactly as the JSONL holds it."""
        ...

    def attached(self, path: Path) -> None:
        """A file the run wrote, kept with the run."""
        ...

    def ended(self) -> None:
        """The run finished its epochs."""
        ...


@dataclass(frozen=True)
class Fanout:
    """Every configured mirror, called in turn; none by default, so an unconfigured run pays nothing."""

    sinks: tuple[Tracker, ...] = ()

    @classmethod
    def configured(cls, environment: Mapping[str, str] = os.environ) -> "Fanout":
        """The mirrors this process asked for: MLflow when `KNEEMRI_MLFLOW` names a store, else none."""
        uri = environment.get(MLFLOW_ENV)
        return cls((MlflowTracker.opened(uri),) if uri else ())

    def begun(self, name: str, recipe: Mapping[str, str], run: Mapping[str, object]) -> None:
        """Forward to every sink."""
        for sink in self.sinks:
            sink.begun(name, recipe, run)

    def logged(self, row: Mapping[str, object]) -> None:
        """Forward to every sink."""
        for sink in self.sinks:
            sink.logged(row)

    def attached(self, path: Path) -> None:
        """Forward to every sink."""
        for sink in self.sinks:
            sink.attached(path)

    def ended(self) -> None:
        """Forward to every sink."""
        for sink in self.sinks:
            sink.ended()
