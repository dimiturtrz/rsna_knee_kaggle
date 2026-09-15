"""The shared recipe exists twice — shell variables every flag arm splices, and the layer a TOML arm stacks.

`experiments/recipe.sh` feeds four hundred flag invocations and `experiments/recipes/base.toml` feeds the
arms written as recipe files. Two spellings of one recipe drift the way knee-7tq did, so this parses both
through the real parser and asks for the same tree.
"""

import re
from pathlib import Path

import pytest

from kneemri.training.config_tree import TrainConfig

REPO = Path(__file__).parents[2]
SHARED = ("RECIPE", "THROUGHPUT")


def test_base_toml_is_the_shell_recipe(monkeypatch: pytest.MonkeyPatch):
    """RECIPE and THROUGHPUT as flags parse to exactly the tree base.toml does as a layer."""
    monkeypatch.chdir(REPO)
    shell = (REPO / "experiments" / "recipe.sh").read_text(encoding="utf-8")
    flags = [token for name in SHARED for token in re.findall(rf'^{name}="([^"]*)"', shell, re.MULTILINE)[0].split()]

    as_flags = TrainConfig.of(["--labels", "l.csv", *flags])
    as_layer = TrainConfig.of(["--labels", "l.csv", "--config", "experiments/recipes/base.toml"])

    assert as_layer.with_leaves(config=None) == as_flags
