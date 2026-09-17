"""A durable write lands whole, and leaves no temp file behind."""

from pathlib import Path

import torch

from kneemri.training.durable import Durable


def test_save(tmp_path: Path) -> None:
    """The payload reloads identically, under the name asked for and with the temp file gone."""
    destination = tmp_path / "weights.partial"
    Durable.save({"model": torch.ones(3), "epoch": 4}, destination)

    held = torch.load(destination, map_location="cpu", weights_only=False)
    assert held["epoch"] == 4
    assert torch.equal(held["model"], torch.ones(3))
    assert list(tmp_path.iterdir()) == [destination]


def test_save_replaces_an_existing_file(tmp_path: Path) -> None:
    """A second write is the file the reader gets — the partial is rewritten every improving epoch."""
    destination = tmp_path / "weights.partial"
    Durable.save({"epoch": 1}, destination)
    Durable.save({"epoch": 2}, destination)

    assert torch.load(destination, map_location="cpu", weights_only=False)["epoch"] == 2
