"""`WeightSoup` averages compatible checkpoints and refuses incompatible ones."""

from pathlib import Path

import pytest
import torch

from kneemri.models.weight_soup import IncompatibleSoupError, WeightSoup
from kneemri.runs.run_record import RunRecord

WIDTH = 4
"""Two tensors wide enough to tell an average from either input, and small enough to write in a test."""


def _options(**overrides: str) -> dict[str, str]:
    """A recipe carrying every key the signature reads, at the dense corner's own values."""
    recipe = {
        "variant": "small",
        "pool": "cls_mean",
        "slice_pooling": "max",
        "slot_pooling": "attend",
        "image_size": "336",
        "slices": "20",
        "whole_stack": "True",
        "unfreeze_last": "6",
        "band_low": "0.1",
        "band_high": "0.9",
        "encoder_layer": "-1",
        "lora_rank": "0",
        "folds": "5",
        "split_seed": "2026",
        "fold": "0",
        "labels": "labels_v2.csv",
        "eval_labels": "None",
    }
    return recipe | overrides


def _checkpoint(path: Path, value: float, **overrides: str) -> Path:
    """A checkpoint whose every weight is one constant, so the average is one arithmetic fact."""
    payload = {
        "model": {"head.weight": torch.full((WIDTH, WIDTH), value), "head.bias": torch.full((WIDTH,), value)},
        "options": _options(**overrides),
        "holdout": 0.5,
        "epoch": 3,
        "fingerprint": [1.0, 2.0],
        "fingerprint_config": {"img": 336},
        "run": {"id": f"run-{value}"},
        "config": {"optim": {"lr": 4e-5}, "witness": {"studies": ["s1"], "predictions": [[value]], "batch": 4}},
    }
    torch.save(payload, path)
    return path


def test_weight_soup_signature(tmp_path: Path) -> None:
    """The ruler is the evaluation table when one is named, and the training table when not."""
    trained = WeightSoup([_checkpoint(tmp_path / "a.pt", 1.0), _checkpoint(tmp_path / "b.pt", 3.0)])
    assert trained.signature(trained.payloads[0])[-1] == "labels_v2.csv"
    recut = WeightSoup(
        [
            _checkpoint(tmp_path / "c.pt", 1.0, labels="labels_v3.csv", eval_labels="D:/x/labels_v2.csv"),
            _checkpoint(tmp_path / "d.pt", 3.0),
        ]
    )
    assert recut.signature(recut.payloads[0])[-1] == "labels_v2.csv"


def test_weight_soup_agreed(tmp_path: Path) -> None:
    """Members sharing everything agree; a differing architecture, split or ruler is named and refused."""
    soup = WeightSoup([_checkpoint(tmp_path / "a.pt", 1.0), _checkpoint(tmp_path / "b.pt", 3.0)])
    assert soup.agreed()[0] == "small"
    for key, value in (("slices", "12"), ("fold", "1"), ("labels", "labels_v9.csv")):
        odd = WeightSoup([tmp_path / "a.pt", _checkpoint(tmp_path / f"{key}.pt", 3.0, **{key: value})])
        with pytest.raises(IncompatibleSoupError, match=key):
            odd.agreed()


def test_weight_soup_averaged(tmp_path: Path) -> None:
    """Every tensor is the mean over members, and a mismatched state dict is refused rather than padded."""
    soup = WeightSoup([_checkpoint(tmp_path / "a.pt", 1.0), _checkpoint(tmp_path / "b.pt", 3.0)])
    averaged = soup.averaged()
    assert set(averaged) == {"head.weight", "head.bias"}
    assert torch.allclose(averaged["head.weight"], torch.full((WIDTH, WIDTH), 2.0))
    assert torch.allclose(averaged["head.bias"], torch.full((WIDTH,), 2.0))

    other = tmp_path / "wide.pt"
    payload = torch.load(tmp_path / "b.pt", map_location="cpu", weights_only=False)
    payload["model"] = {"head.weight": torch.zeros(WIDTH, WIDTH + 1), "head.bias": torch.zeros(WIDTH)}
    torch.save(payload, other)
    with pytest.raises(IncompatibleSoupError, match="shape"):
        WeightSoup([tmp_path / "a.pt", other]).averaged()


def test_weight_soup_write(tmp_path: Path) -> None:
    """The soup keeps its first member's recipe, clears its claims, and drops every borrowed witness of it.

    Both claims about a specific set of tensors go: the fingerprint beside the recipe and the end-to-end
    witness nested inside it. Carrying the latter made every soup read as a reading defect rather than as
    a network nobody has witnessed yet.
    """
    soup = WeightSoup([_checkpoint(tmp_path / "a.pt", 1.0), _checkpoint(tmp_path / "b.pt", 3.0)])
    written = torch.load(soup.write(tmp_path / "soup.pt"), map_location="cpu", weights_only=False)
    assert written["options"]["fold"] == "0"
    assert written["soup"] == ["a", "b"]
    assert written["epoch"] < 0
    assert written["holdout"] != written["holdout"]
    assert "fingerprint" not in written
    assert "witness" not in written["config"], "a witness belongs to the tensors that produced it"
    assert written["config"]["optim"] == {"lr": 4e-5}, "the rest of the recipe survives"
    assert torch.allclose(written["model"]["head.bias"], torch.full((WIDTH,), 2.0))
    assert written["run"]["id"] not in {"run-1.0", "run-3.0"}, "a soup is a run of its own, not its first member"
    assert [parent["run"] for parent in written["run"]["parents"].values()] == ["run-1.0", "run-3.0"]
    assert RunRecord(tmp_path / "soup.pt").run() == written["run"], "the record beside it says the same"


def test_weight_soup_refuses_one_member(tmp_path: Path) -> None:
    """One checkpoint averaged with itself is that checkpoint, and a full-data fit has seen the holdout."""
    with pytest.raises(IncompatibleSoupError, match="at least"):
        WeightSoup([_checkpoint(tmp_path / "a.pt", 1.0)])
    members = [_checkpoint(tmp_path / "dark.pt", 1.0, fold="-1"), _checkpoint(tmp_path / "e.pt", 3.0, fold="-1")]
    with pytest.raises(IncompatibleSoupError, match="full-data"):
        WeightSoup(members).agreed()
    shipped = WeightSoup(members, ship_only=True)
    assert shipped.agreed()[-2] == "-1", "an all-full-data pot has no holdout to leak into and ships"
    mixed = [members[0], _checkpoint(tmp_path / "f.pt", 3.0)]
    with pytest.raises(IncompatibleSoupError, match="fold"):
        WeightSoup(mixed, ship_only=True).agreed()
