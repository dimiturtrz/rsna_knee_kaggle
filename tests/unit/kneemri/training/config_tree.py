"""The tree is the one declaration of every knob; the parser, the file reader and the flat view all read it."""

import dataclasses
from pathlib import Path

import pytest

from kneemri.training.augmentation import RegularisationConfig
from kneemri.training.config_tree import NOT_STATED, TrainConfig
from kneemri.training.options import TrainingOptions


def test_leaves():
    """Every leaf is reached exactly once, and no node is mistaken for a leaf."""
    names = [leaf.name for leaf, _ in TrainConfig.leaves()]

    assert len(names) == len(set(names)), "two nodes declare one knob"
    assert "mixup" in names
    assert "regularisation" not in names


def test_parser():
    """Every field is reachable from argv, and every flag lands on a field.

    A field with no flag is a knob only a test can turn; a flag with no field would raise on `parse`.
    """
    dests = set(vars(TrainConfig.parser().parse_args(["--labels", "labels.csv"])))

    assert dests == {field.name for field in dataclasses.fields(TrainingOptions)}


def test_of(tmp_path: Path):
    """Flags land in their node, and a flag on the line beats the recipe file."""
    recipe = tmp_path / "r.toml"
    recipe.write_text("epochs = 36\nmixup = 0.2\n", encoding="utf-8")

    tree = TrainConfig.of(["--labels", "l.csv", "--config", str(recipe), "--epochs", "4", "--eager-encoder"])

    assert tree.optim.epochs == 4
    assert tree.regularisation.mixup == pytest.approx(0.2)
    assert tree.run.compile_encoder is False, "a renamed flag still lands on its field"


def test_of_flat():
    """A mapping names leaves; an unnamed leaf keeps its default."""
    tree = TrainConfig.of_flat({"mixup": 0.3, "epochs": 7})

    assert tree.regularisation == dataclasses.replace(RegularisationConfig(), mixup=0.3)
    assert tree.optim.epochs == 7


def test_flat():
    """The flat view round-trips through `of_flat` and holds every leaf."""
    tree = TrainConfig.of_flat({"slices": 20, "variant": "base"})

    assert TrainConfig.of_flat(tree.flat) == tree
    assert tree.flat["slices"] == 20
    assert len(tree.flat) == len(list(TrainConfig.leaves()))


def test_of_file(tmp_path: Path):
    """Values arrive typed as their fields; no file is no values; a misspelt key refuses."""
    recipe = tmp_path / "r.toml"
    recipe.write_text('warmup = 0.25\nslices = 20\npadding_mode = "reflection"\nunread_targets = "u.npz"\n', "utf-8")

    assert TrainConfig.of_file(recipe) == {
        "warmup": 0.25,
        "slices": 20,
        "padding_mode": "reflection",
        "unread_targets": Path("u.npz"),
    }
    assert TrainConfig.of_file(None) == {}
    recipe.write_text("stochastic_dept = 0.05\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stochastic_dept"):
        TrainConfig.of_file(recipe)


def test_typed():
    """Text reads as the declared type, an optional reads its stated half, and the unstated marker is None."""
    assert TrainConfig.typed(int, "3") == 3
    assert TrainConfig.typed(Path | None, "a.npz") == Path("a.npz")
    assert TrainConfig.typed(bool, "True") is True
    assert TrainConfig.typed(float | None, NOT_STATED) is None
