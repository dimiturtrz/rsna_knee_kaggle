import inspect

import pytest

from kneemri.runs.mlflow_client import MlflowClient


def test_mlflow_client_names_only_what_mlflow_has():
    """Every method the protocol names exists on the real client, so a tracker never calls a method mlflow lacks."""
    tracking = pytest.importorskip("mlflow.tracking")
    named = {name for name, _ in inspect.getmembers(MlflowClient, inspect.isfunction) if not name.startswith("_")}

    assert named
    assert named <= set(dir(tracking.MlflowClient))
