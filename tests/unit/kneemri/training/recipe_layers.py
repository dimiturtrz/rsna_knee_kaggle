from pathlib import Path

import pytest

from kneemri.training.recipe_layers import RecipeLayers

ADDRESSES = {"optim.epochs": "epochs", "optim.ema": "ema", "regularisation.mixup": "mixup", "run.labels": "labels"}


def test_of():
    """Every `--config` and `--set` is kept in line order; a flag with no value after it is not a layer."""
    layers = RecipeLayers.of(["--config", "a.toml", "--epochs", "4", "--set", "optim.ema=true", "--config", "b.toml"])

    assert layers == RecipeLayers((Path("a.toml"), Path("b.toml")), ("optim.ema=true",))
    assert RecipeLayers.of(["--labels", "l.csv", "--config"]) == RecipeLayers()


def test_values(tmp_path: Path):
    """Nested tables read by dotted path, a later file over an earlier, `--set` over both, raw values kept."""
    first = tmp_path / "a.toml"
    first.write_text("[optim]\nepochs = 36\nema = false\n[regularisation]\nmixup = 0.1\n", encoding="utf-8")
    second = tmp_path / "b.toml"
    second.write_text("[optim]\nema = true\n", encoding="utf-8")

    values = RecipeLayers((first, second), ("regularisation.mixup=0.3", "run.labels=d/l.csv")).values(ADDRESSES)

    assert values == {"epochs": 36, "ema": True, "mixup": 0.3, "labels": "d/l.csv"}
    assert RecipeLayers().values(ADDRESSES) == {}


def test_values_refuses_a_path_naming_no_leaf(tmp_path: Path):
    """A misspelt key, and a real knob filed under the wrong node, both refuse instead of reading as nothing."""
    recipe = tmp_path / "r.toml"
    recipe.write_text("[optim]\nmixup = 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"optim\.mixup"):
        RecipeLayers((recipe,)).values(ADDRESSES)
    with pytest.raises(ValueError, match="stochastic_dept"):
        RecipeLayers(assignments=("regularisation.stochastic_dept=0.05",)).values(ADDRESSES)
