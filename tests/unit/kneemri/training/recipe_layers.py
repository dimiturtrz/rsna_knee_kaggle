from pathlib import Path

import pytest

from core.paths import ENV_OVERRIDE, DataRoot
from kneemri.training.recipe_layers import SET, RecipeLayers, RecipeMeta, RecipeStatus

ADDRESSES = {"optim.epochs": "epochs", "optim.ema": "ema", "regularisation.mixup": "mixup", "run.labels": "labels"}


def test_recipe_layers_of():
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


def test_recipe_meta_of():
    """`status` reads as the enum, `requires` and `after` as tuples, and a key naming no field refuses."""
    step = "kneemri.evaluation.ledger75 --only x_s2026"
    meta = RecipeMeta.of({"status": "done", "requires": ["a.npz", "b.npz"], "after": [step], "out": "x_s2026"})

    expected = RecipeMeta(status=RecipeStatus.DONE, requires=("a.npz", "b.npz"), after=(step,), out="x_s2026")
    assert meta == expected
    assert RecipeMeta.of({}) == RecipeMeta()
    with pytest.raises(ValueError, match="hypotesis"):
        RecipeMeta.of({"hypotesis": "typo"})


def test_sources(tmp_path: Path):
    """Each leaf carries the layer that set it last: a parent, the file extending it, or `--set`."""
    parent = tmp_path / "trunk.toml"
    parent.write_text("[optim]\nepochs = 12\nema = false\n", encoding="utf-8")
    leaf = tmp_path / "family" / "leaf.toml"
    leaf.parent.mkdir()
    leaf.write_text('extends = ["../trunk.toml"]\n[optim]\nema = true\n[meta]\nout = "leaf_s2026"\n', encoding="utf-8")

    sources = RecipeLayers((leaf,), ("regularisation.mixup=0.2",)).sources(ADDRESSES)

    assert sources == {"epochs": (12, str(parent)), "ema": (True, str(leaf)), "mixup": (0.2, SET)}


def test_sources_fill_a_data_root_from_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A file names its input under `$PROCESSED`, the root `lib.sh` exports, never a machine path."""
    monkeypatch.setenv("PROCESSED", "/data/processed")
    monkeypatch.setenv("MODELS", "/data/models")
    leaf = tmp_path / "leaf.toml"
    leaf.write_text('[run]\nlabels = "$PROCESSED/labels/l.csv"\n[optim]\nepochs = 4\n', encoding="utf-8")

    sources = RecipeLayers((leaf,)).sources(ADDRESSES)

    assert sources == {"labels": ("/data/processed/labels/l.csv", str(leaf)), "epochs": (4, str(leaf))}


def test_sources_refuse_a_root_the_environment_does_not_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An unfilled `$MODELS` would train against a path that exists nowhere, found only once the card is held."""
    monkeypatch.chdir(tmp_path)
    for name in ("MODELS", "PROCESSED", ENV_OVERRIDE):
        monkeypatch.delenv(name, raising=False)
    leaf = tmp_path / "leaf.toml"
    leaf.write_text('[run]\nlabels = "$MODELS/x.csv"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="does not set"):
        RecipeLayers((leaf,)).sources(ADDRESSES)


def test_sources_fill_a_data_root_from_paths_yaml_outside_an_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A reader outside `lib.sh` (arm_check, a review) resolves a root where an arm would: from `paths.yaml`."""
    monkeypatch.chdir(tmp_path)
    for name in ("MODELS", "PROCESSED", ENV_OVERRIDE):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "paths.yaml").write_text(f"data: {(tmp_path / 'root').as_posix()}\n", encoding="utf-8")
    leaf = tmp_path / "leaf.toml"
    leaf.write_text('[run]\nlabels = "$MODELS/x.csv"\n', encoding="utf-8")

    sources = RecipeLayers((leaf,)).sources(ADDRESSES)

    models = DataRoot.from_config(Path("paths.yaml")).models().as_posix()
    assert sources == {"labels": (f"{models}/x.csv", str(leaf))}


def test_values_expands_extends_depth_first_with_parents_first(tmp_path: Path):
    """A grandparent, then a parent, then the leaf; each later layer winning over the one it extends."""
    base = "[optim]\nepochs = 36\nema = false\n[regularisation]\nmixup = 0.1\n"
    (tmp_path / "base.toml").write_text(base, encoding="utf-8")
    (tmp_path / "trunk.toml").write_text('extends = ["base.toml"]\n[optim]\nepochs = 12\n', encoding="utf-8")
    leaf = tmp_path / "leaf.toml"
    leaf.write_text('extends = ["trunk.toml"]\n[regularisation]\nmixup = 0.0\n', encoding="utf-8")

    assert RecipeLayers((leaf,)).values(ADDRESSES) == {"epochs": 12, "ema": False, "mixup": 0.0}


def test_values_refuses_a_file_that_extends_itself(tmp_path: Path):
    """A cycle through `extends` refuses, naming the chain, instead of recursing until the stack runs out."""
    (tmp_path / "a.toml").write_text('extends = ["b.toml"]\n', encoding="utf-8")
    (tmp_path / "b.toml").write_text('extends = ["a.toml"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="extends itself"):
        RecipeLayers((tmp_path / "a.toml",)).values(ADDRESSES)


def test_meta(tmp_path: Path):
    """The `[meta]` down the chain merges with the leaf winning, and is never read as a training option."""
    family = tmp_path / "_family.toml"
    family.write_text('[meta]\nstatus = "done"\nbead = "knee-family"\ncontrol = "c.sh"\n', encoding="utf-8")
    leaf = tmp_path / "leaf.toml"
    leaf.write_text('extends = ["_family.toml"]\n[optim]\nepochs = 4\n[meta]\nbead = "knee-leaf"\n', encoding="utf-8")

    layers = RecipeLayers((leaf,))

    assert layers.meta() == RecipeMeta(status=RecipeStatus.DONE, bead="knee-leaf", control="c.sh")
    assert layers.values(ADDRESSES) == {"epochs": 4}
    assert RecipeLayers().meta() == RecipeMeta()
