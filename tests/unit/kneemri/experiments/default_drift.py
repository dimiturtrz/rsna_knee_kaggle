"""A parent's recorded settings against today's defaults — the pin list a one-knob arm needs."""

import json
from pathlib import Path

from kneemri.experiments.default_drift import DefaultDrift
from kneemri.training.config_tree import TrainConfig


def _recipe() -> dict[str, str]:
    """The defaults exactly as a run that started this second would record them."""
    return dict(TrainConfig().recipe)


def test_declared() -> None:
    """The defaults carry the IDENTITY values an off-list cannot express — 1.0 is not 0, and not absent."""
    defaults = DefaultDrift.declared()
    assert defaults["layer_decay"] == "1.0"
    assert defaults["accumulate"] == "1"


def test_drifted() -> None:
    """No drift is a measurement, and the check must be able to say it."""
    assert DefaultDrift(_recipe()).drifted() == {}


def test_drifted_names_the_moved_key_with_both_values() -> None:
    """The pin list is the pair: what the parent got, and what its child would get instead."""
    recipe = _recipe()
    recipe["weight_decay"] = "0.99"
    assert DefaultDrift(recipe).drifted() == {"weight_decay": ("0.99", TrainConfig().recipe["weight_decay"])}


def test_drifted_skips_an_incidental_key() -> None:
    """Every arm names its own checkpoint, so `out` disagreeing with the default is not drift."""
    recipe = _recipe()
    recipe["out"] = "somewhere/else.pt"
    assert "out" not in DefaultDrift(recipe).drifted()


def test_drifted_skips_a_key_the_recorded_row_never_carried() -> None:
    """A field added after the parent ran ARRIVED; it did not drift, and cannot be pinned at absent."""
    recipe = _recipe()
    del recipe["epochs"]
    assert "epochs" not in DefaultDrift(recipe).drifted()


def test_of_metrics(tmp_path: Path) -> None:
    """Row 0 is the recipe row, and it is the only line this reads."""
    recipe = _recipe()
    recipe["warmup"] = "0.42"
    metrics = tmp_path / "parent.metrics.jsonl"
    metrics.write_text(
        json.dumps({"recipe": recipe}) + "\n" + json.dumps({"epoch": 0, "auc": 0.5}) + "\n", encoding="utf-8"
    )
    assert DefaultDrift.of_metrics(metrics).drifted()["warmup"][0] == "0.42"
